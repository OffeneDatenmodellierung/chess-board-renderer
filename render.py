"""Render labelled photos of chess boards with Blender, for training a board
reader (open-chess's ``chess-vision-train``, dataset format ``jsonl``).

Everything in a render is made here: the pieces are modelled from profiles
in this file, the boards, tables, lights and clutter are generated, and the
positions come from ``positions.py``. No outside models, textures or photos
go in, so the images are entirely yours to license.

    blender -b -P render.py -- --positions positions.txt --out renders --count 1000

or, with Blender as a Python module (Python 3.11):

    uv run --extra blender render.py --positions positions.txt --out renders --count 1000

Writes ``renders/photos/000000.jpg`` … and ``renders/labels.jsonl``, one line
per photo: ``{"image": "photos/000000.jpg", "placement": "<FEN placement>",
"group": "pos-000123"}``. Labels are from White's side whichever side the
camera is on (as the trainer wants); views of one position share a group, so
they never straddle a train/validation split. A run can be stopped and
started again: it carries on from the last photo written.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

# --------------------------------------------------------------- positions

PIECES = "PNBRQKpnbrqk"


def parse_placement(fen: str) -> list[tuple[int, int, str]]:
    """(file, rank, piece) for each piece of a FEN placement, a1 = (0, 0)."""
    rows = fen.split()[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"not a board: {fen!r}")
    out = []
    for i, row in enumerate(rows):
        rank, file = 7 - i, 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            elif ch in PIECES:
                out.append((file, rank, ch))
                file += 1
            else:
                raise ValueError(f"not a board: {fen!r}")
        if file != 8:
            raise ValueError(f"not a board: {fen!r}")
    return out


# ------------------------------------------------------------------ meshes


def lathe(profile, segments=40):
    """A solid of revolution about z from (radius, height) points, bottom to
    top; points with radius 0 close the ends."""
    verts, rings = [], []
    for r, z in profile:
        if r <= 1e-6:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ring = []
            for s in range(segments):
                a = 2 * math.pi * s / segments
                ring.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ring)
    faces = []
    for lo, hi in zip(rings, rings[1:]):
        if len(lo) == 1 and len(hi) == 1:
            continue
        if len(lo) == 1:
            faces += [(lo[0], hi[(s + 1) % segments], hi[s]) for s in range(segments)]
        elif len(hi) == 1:
            faces += [(lo[s], lo[(s + 1) % segments], hi[0]) for s in range(segments)]
        else:
            faces += [(lo[s], lo[(s + 1) % segments], hi[(s + 1) % segments], hi[s]) for s in range(segments)]
    return verts, faces


def box(cx, cy, z0, w, d, h, rot=0.0):
    """A box centred on (cx, cy), from z0 up h, turned rot about z."""
    c, s = math.cos(rot), math.sin(rot)
    pts = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    verts = [(cx + x * c - y * s, cy + x * s + y * c, z) for z in (z0, z0 + h) for x, y in pts]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, faces


def ball(cx, cy, cz, r, segs=12):
    prof = [(r * math.sin(math.pi * i / segs), cz - r * math.cos(math.pi * i / segs)) for i in range(segs + 1)]
    v, f = lathe(prof, 16)
    return [(x + cx, y + cy, z) for x, y, z in v], f


def extrude(outline, thickness):
    """A polygon in the (y, z) plane, thickened along x."""
    n = len(outline)
    t = thickness / 2
    verts = [(-t, y, z) for y, z in outline] + [(t, y, z) for y, z in outline]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return verts, faces


def join(parts):
    verts, faces = [], []
    for v, f in parts:
        base = len(verts)
        verts += v
        faces += [tuple(i + base for i in face) for face in f]
    return verts, faces


def piece_geometry(kind: str, style: dict):
    """One piece type's mesh, in square units (a square is 1 across)."""
    rs, zs = style["radius"], style["height"]

    def prof(points):
        return [(r * rs, z * zs) for r, z in points]

    foot = [(0, 0), (1.0, 0), (1.0, 0.06), (0.86, 0.10), (0.74, 0.16)]
    if kind == "p":
        b = 0.30
        p = [(r * b, z) for r, z in foot] + [
            (0.14, 0.22), (0.10, 0.40), (0.17, 0.44), (0.17, 0.47), (0.09, 0.50),
            (0.13, 0.55), (0.155, 0.62), (0.13, 0.70), (0.07, 0.75), (0, 0.765),
        ]
        return join([lathe(prof(p))])
    if kind == "r":
        b = 0.33
        p = [(r * b, z) for r, z in foot] + [
            (0.17, 0.30), (0.17, 0.62), (0.245, 0.66), (0.245, 0.84), (0.17, 0.84), (0.17, 0.79), (0, 0.79),
        ]
        top = 0.84 * zs
        merlons = [
            box(0.205 * rs * math.cos(a), 0.205 * rs * math.sin(a), top, 0.075 * rs, 0.13 * rs, 0.07 * zs, a)
            for a in (math.pi / 4 + k * math.pi / 2 for k in range(4))
        ]
        return join([lathe(prof(p))] + merlons)
    if kind == "b":
        b = 0.32
        p = [(r * b, z) for r, z in foot] + [
            (0.13, 0.35), (0.10, 0.62), (0.19, 0.66), (0.19, 0.69), (0.10, 0.72), (0.15, 0.80),
            (0.165, 0.90), (0.12, 1.00), (0.06, 1.06), (0.045, 1.09), (0.06, 1.12), (0.035, 1.15), (0, 1.16),
        ]
        return join([lathe(prof(p))])
    if kind == "q":
        b = 0.35
        p = [(r * b, z) for r, z in foot] + [
            (0.14, 0.42), (0.11, 0.80), (0.22, 0.86), (0.22, 0.89), (0.12, 0.92), (0.17, 1.10),
            (0.21, 1.19), (0.15, 1.22), (0.10, 1.24), (0.06, 1.30), (0, 1.31),
        ]
        crown = [
            ball(0.19 * rs * math.cos(a), 0.19 * rs * math.sin(a), 1.215 * zs, 0.035 * rs)
            for a in (k * math.pi / 4 for k in range(8))
        ]
        return join([lathe(prof(p)), ball(0, 0, 1.34 * zs, 0.05 * rs)] + crown)
    if kind == "k":
        b = 0.36
        p = [(r * b, z) for r, z in foot] + [
            (0.15, 0.44), (0.12, 0.84), (0.23, 0.90), (0.23, 0.93), (0.13, 0.96), (0.18, 1.16),
            (0.20, 1.26), (0.13, 1.30), (0.10, 1.32), (0.10, 1.36), (0, 1.37),
        ]
        z0 = 1.37 * zs
        cross = [box(0, 0, z0, 0.055 * rs, 0.055 * rs, 0.20 * zs), box(0, 0, z0 + 0.10 * zs, 0.17 * rs, 0.055 * rs, 0.05 * zs)]
        return join([lathe(prof(p))] + cross)
    if kind == "n":
        b = 0.33
        base = [(r * b, z) for r, z in foot] + [(0.22, 0.20), (0.20, 0.25), (0, 0.25)]
        # The head's side view, facing +y, from the base up and round.
        head = [
            (-0.17, 0.24), (0.15, 0.24), (0.11, 0.38), (0.21, 0.58), (0.27, 0.70), (0.26, 0.76),
            (0.20, 0.80), (0.08, 0.77), (0.04, 0.86), (0.00, 0.95), (-0.05, 0.90), (-0.12, 0.84),
            (-0.18, 0.72), (-0.21, 0.56), (-0.20, 0.40),
        ]
        head = [(y * rs, z * zs) for y, z in head]
        return join([lathe(prof(base)), extrude(head, 0.22 * rs)])
    raise ValueError(kind)


def mesh_from(name, geom, smooth=True):
    verts, faces = geom
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


# --------------------------------------------------------------- materials


def srgb(c):
    return tuple(x**2.2 for x in c)


def jitter(rng, c, amount=0.06):
    h, l, s = colorsys.rgb_to_hls(*c)
    l = min(max(l + rng.uniform(-amount, amount), 0.0), 1.0)
    s = min(max(s * rng.uniform(0.8, 1.2), 0.0), 1.0)
    h = (h + rng.uniform(-0.02, 0.02)) % 1.0
    return colorsys.hls_to_rgb(h, l, s)


def material(name, colour, rng, roughness=0.5, grain=0.0, metallic=0.0):
    """A Principled material, optionally with wood-like grain (a wave
    texture between the colour and a darker one)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    lin = srgb(colour)
    bsdf.inputs["Base Color"].default_value = (*lin, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if grain > 0:
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value = rng.uniform(1.5, 6.0)
        wave.inputs["Distortion"].default_value = rng.uniform(2.0, 10.0)
        wave.inputs["Detail"].default_value = 3.0
        mapping = nt.nodes.new("ShaderNodeMapping")
        coords = nt.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value = (rng.uniform(0, 3.14), rng.uniform(0, 3.14), rng.uniform(0, 3.14))
        mapping.inputs["Location"].default_value = (rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50))
        nt.links.new(coords.outputs["Object"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*lin, 1.0)
        dark = tuple(x * (1 - grain) for x in lin)
        ramp.color_ramp.elements[1].color = (*dark, 1.0)
        nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


# Light and dark squares (sRGB).
BOARDS = [
    ((0.82, 0.66, 0.44), (0.45, 0.27, 0.14), 0.35),  # wood
    ((0.93, 0.92, 0.80), (0.45, 0.58, 0.34), 0.0),  # tournament green
    ((0.92, 0.92, 0.88), (0.08, 0.08, 0.08), 0.0),  # black and white
    ((0.86, 0.89, 0.93), (0.30, 0.42, 0.62), 0.0),  # blue
    ((0.78, 0.76, 0.72), (0.38, 0.36, 0.34), 0.0),  # grey
    ((0.93, 0.85, 0.70), (0.62, 0.42, 0.24), 0.2),  # light wood
    ((0.95, 0.90, 0.75), (0.55, 0.20, 0.15), 0.0),  # cream and red
]
# White and black pieces (sRGB), roughness, grain.
SETS = [
    ((0.88, 0.75, 0.52), (0.07, 0.05, 0.04), 0.35, 0.25),  # boxwood and ebony
    ((0.93, 0.91, 0.85), (0.04, 0.04, 0.04), 0.25, 0.0),  # plastic
    ((0.88, 0.78, 0.62), (0.38, 0.14, 0.09), 0.40, 0.25),  # boxwood and rosewood
    ((0.95, 0.93, 0.88), (0.12, 0.10, 0.09), 0.55, 0.0),  # matt plastic
    ((0.92, 0.86, 0.72), (0.25, 0.15, 0.08), 0.30, 0.3),  # natural and stained wood
]
TABLES = [
    ((0.55, 0.38, 0.22), 0.4),
    ((0.75, 0.62, 0.45), 0.3),
    ((0.85, 0.85, 0.83), 0.0),
    ((0.20, 0.20, 0.22), 0.0),
    ((0.35, 0.45, 0.30), 0.0),
    ((0.60, 0.20, 0.15), 0.0),
    ((0.40, 0.30, 0.25), 0.5),
]


# ------------------------------------------------------------------- scene


class Scene:
    """The parts of the scene that change: rebuilt as needed between photos."""

    def __init__(self, rng: random.Random, args):
        self.rng = rng
        self.args = args
        self.piece_meshes: dict[str, bpy.types.Mesh] = {}
        self.dynamic: list[bpy.types.Object] = []
        self.style_count = 0
        self.scene = bpy.context.scene
        self.setup_render()
        self.camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
        self.scene.collection.objects.link(self.camera)
        self.scene.camera = self.camera
        self.camera.data.clip_start = 0.1
        self.camera.data.clip_end = 400

    def setup_render(self):
        s = self.scene
        s.render.engine = "CYCLES"
        s.cycles.samples = self.args.samples
        s.cycles.use_denoising = True
        s.cycles.max_bounces = 4
        s.render.image_settings.file_format = "JPEG"
        s.render.image_settings.quality = 92
        s.render.film_transparent = False
        if not self.args.cpu:
            enable_gpu(s)

    def link(self, ob):
        self.scene.collection.objects.link(ob)
        self.dynamic.append(ob)
        return ob

    def clear(self):
        for ob in self.dynamic:
            bpy.data.objects.remove(ob, do_unlink=True)
        self.dynamic = []
        for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.images):
            for block in list(coll):
                if block.users == 0:
                    coll.remove(block)

    def new_set_style(self):
        """A new piece design (proportions)."""
        rng = self.rng
        # The old design's meshes go once nothing uses them (see clear()).
        for me in self.piece_meshes.values():
            me.use_fake_user = False
        style = {"radius": rng.uniform(0.85, 1.12), "height": rng.uniform(0.85, 1.12) * rng.uniform(0.95, 1.1)}
        self.piece_meshes = {}
        for k in "pnbrqk":
            me = mesh_from(f"piece-{k}", piece_geometry(k, style), smooth=k != "n")
            # Kept between photos, and coloured per object (one slot).
            me.use_fake_user = True
            me.materials.append(None)
            self.piece_meshes[k] = me

    def build(self, placement: str):
        rng = self.rng
        self.clear()
        if self.style_count % max(self.args.style_every, 1) == 0 or not self.piece_meshes:
            self.new_set_style()
        self.style_count += 1

        # World light.
        world = self.scene.world or bpy.data.worlds.new("world")
        self.scene.world = world
        if hasattr(world, "use_nodes") and not world.use_nodes:
            world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        tint = colorsys.hls_to_rgb(rng.uniform(0.05, 0.6), 0.5, rng.uniform(0.0, 0.25))
        bg.inputs["Color"].default_value = (*srgb(tint), 1.0)
        bg.inputs["Strength"].default_value = rng.uniform(0.3, 1.2)
        # Phones expose for the scene: a little brighter or darker.
        self.scene.view_settings.exposure = rng.uniform(0.2, 1.2)

        # Table.
        tcol, tgrain = rng.choice(TABLES)
        table = self.link(bpy.data.objects.new("table", mesh_from("table", box(0, 0, -0.3, 80, 80, 0.15), False)))
        table.data.materials.append(material("table", jitter(rng, tcol, 0.1), rng, rng.uniform(0.3, 0.9), tgrain))

        # Board: 64 squares (a1 dark) on a frame.
        light, dark, grain = rng.choice(BOARDS)
        light, dark = jitter(rng, light), jitter(rng, dark)
        rough = rng.uniform(0.2, 0.7)
        verts, faces, mats = [], [], []
        for f in range(8):
            for r in range(8):
                x0, y0 = f - 4, r - 4
                i = len(verts)
                verts += [(x0, y0, 0), (x0 + 1, y0, 0), (x0 + 1, y0 + 1, 0), (x0, y0 + 1, 0)]
                faces.append((i, i + 1, i + 2, i + 3))
                mats.append(1 if (f + r) % 2 == 0 else 0)
        board = self.link(bpy.data.objects.new("board", mesh_from("board", (verts, faces), False)))
        board.data.materials.append(material("light", light, rng, rough, grain))
        board.data.materials.append(material("dark", dark, rng, rough, grain))
        board.data.polygons.foreach_set("material_index", mats)
        self.border = rng.choice([0.0, rng.uniform(0.15, 0.9)]) if rng.random() < 0.3 else rng.uniform(0.15, 0.9)
        b = self.border
        thick = rng.uniform(0.05, 0.25)
        frame = self.link(
            bpy.data.objects.new("frame", mesh_from("frame", box(0, 0, -thick - 0.002, 8 + 2 * b, 8 + 2 * b, thick), False))
        )
        fcol = jitter(rng, rng.choice([dark, light, (0.30, 0.18, 0.10), (0.1, 0.1, 0.1)]))
        frame.data.materials.append(material("frame", fcol, rng, rough, grain))
        # The table sits under the board.
        table.location.z = -thick + 0.148

        # Pieces.
        wcol, bcol, prough, pgrain = rng.choice(SETS)
        wmat = material("white", jitter(rng, wcol, 0.04), rng, prough * rng.uniform(0.7, 1.3), pgrain)
        bmat = material("black", jitter(rng, bcol, 0.03), rng, prough * rng.uniform(0.7, 1.3), pgrain)
        scale = rng.uniform(0.85, 1.05)
        wobble = rng.uniform(0.0, 0.12)
        for f, r, ch in parse_placement(placement):
            self.add_piece(ch, f - 3.5 + rng.uniform(-wobble, wobble), r - 3.5 + rng.uniform(-wobble, wobble), scale, wmat, bmat)

        # Captured pieces beside the board now and then (not on a square, so
        # not in the label).
        if rng.random() < 0.35:
            for _ in range(rng.randint(1, 6)):
                ch = rng.choice(PIECES[:-1].replace("K", ""))
                side = rng.choice([-1, 1])
                x = side * (4 + b + rng.uniform(0.6, 1.6))
                self.add_piece(ch, x, rng.uniform(-4, 4), scale, wmat, bmat)

        # Clutter on the table.
        for k in range(rng.choice([0, 0, 1, 2, 3])):
            ang = rng.uniform(0, 2 * math.pi)
            dist = 4 + b + rng.uniform(2.0, 7.0)
            x, y = dist * math.cos(ang), dist * math.sin(ang)
            if rng.random() < 0.5:
                rad, h = rng.uniform(0.3, 0.7), rng.uniform(0.5, 1.8)
                geom = lathe([(0, 0), (rad, 0), (rad * 1.1, h), (0, h)], 32)
            else:
                geom = box(0, 0, 0, rng.uniform(0.5, 3), rng.uniform(0.5, 3), rng.uniform(0.05, 1.2), 0)
            ob = self.link(bpy.data.objects.new(f"clutter{k}", mesh_from("clutter", geom)))
            ob.location = (x, y, 0)
            ob.rotation_euler.z = rng.uniform(0, 2 * math.pi)
            hue = colorsys.hls_to_rgb(rng.random(), rng.uniform(0.2, 0.8), rng.uniform(0.0, 0.7))
            ob.data.materials.append(material("clutter", hue, rng, rng.uniform(0.1, 0.9)))

        self.add_lights()

    def add_piece(self, ch, x, y, scale, wmat, bmat):
        rng = self.rng
        kind = ch.lower()
        white = ch.isupper()
        ob = self.link(bpy.data.objects.new(ch, self.piece_meshes[kind]))
        ob.location = (x, y, 0)
        ob.scale = (scale, scale, scale)
        if kind == "n":
            # Knights face the other side, give or take.
            face = 0.0 if white else math.pi
            ob.rotation_euler.z = face + rng.gauss(0, 0.5)
        else:
            ob.rotation_euler.z = rng.uniform(0, 2 * math.pi)
        # Per-object material: the mesh is shared by every piece of a kind.
        ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = wmat if white else bmat

    def add_lights(self):
        rng = self.rng
        n = rng.choice([1, 1, 2, 2, 3])
        for i in range(n):
            if rng.random() < 0.25:
                light = bpy.data.lights.new(f"sun{i}", "SUN")
                light.energy = rng.uniform(1.0, 4.0)
                light.angle = rng.uniform(0.02, 0.4)
            else:
                light = bpy.data.lights.new(f"area{i}", "AREA")
                light.size = rng.uniform(0.5, 8.0)
            warm = rng.uniform(-1, 1)
            light.color = (1.0, 1.0 - 0.15 * max(warm, 0) + 0.05 * min(warm, 0), 1.0 - 0.35 * max(warm, 0) + 0.1 * min(warm, 0))
            ob = self.link(bpy.data.objects.new(f"light{i}", light))
            az, el = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(25, 85))
            dist = rng.uniform(6, 22)
            ob.location = (dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az), dist * math.sin(el))
            ob.rotation_euler = (-ob.location).to_track_quat("-Z", "Y").to_euler()
            if light.type == "AREA":
                light.energy = rng.uniform(150, 900) * (dist / 10) ** 2 / n

    def place_camera(self, width, height) -> bool:
        """A random view with the whole board in frame; False if none was
        found."""
        rng = self.rng
        cam = self.camera
        s = self.scene
        s.render.resolution_x, s.render.resolution_y = width, height
        cam.data.sensor_fit = "AUTO"
        b = self.border
        corners = [Vector((x, y, 0)) for x in (-4 - b, 4 + b) for y in (-4 - b, 4 + b)]
        for _ in range(40):
            cam.data.lens = rng.uniform(22, 55)
            az = rng.uniform(0, 2 * math.pi)
            el = math.radians(rng.uniform(22, 82))
            target = Vector((rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), 0))
            roll = math.radians(rng.gauss(0, 5))
            fill = rng.uniform(0.55, 0.97)
            direction = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))

            def extent(d):
                cam.location = target + direction * d
                quat = (-direction).to_track_quat("-Z", "Y")
                cam.rotation_euler = (quat.to_matrix() @ Matrix.Rotation(roll, 3, "Z")).to_euler()
                bpy.context.view_layer.update()
                pts = [world_to_camera_view(s, cam, c) for c in corners]
                if any(p.z <= 0.1 for p in pts):
                    return None
                return pts

            lo, hi = 4.0, 120.0
            for _ in range(30):
                mid = (lo + hi) / 2
                pts = extent(mid)
                if pts is None:
                    lo = mid
                    continue
                span = max(max(abs(p.x - 0.5), abs(p.y - 0.5)) for p in pts) * 2
                if span > fill:
                    lo = mid
                else:
                    hi = mid
            pts = extent(hi)
            if pts is None:
                continue
            m = 0.02
            if all(m <= p.x <= 1 - m and m <= p.y <= 1 - m for p in pts):
                cam.data.dof.use_dof = rng.random() < 0.15
                if cam.data.dof.use_dof:
                    cam.data.dof.focus_distance = (cam.location - target).length
                    cam.data.dof.aperture_fstop = rng.uniform(4, 16)
                return True
        return False


def enable_gpu(scene):
    """Cycles on the GPU where there is one (Metal on a Mac)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return
    for kind in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print(f"rendering on {kind}: {', '.join(d.name for d in devices)}")
            return
    print("rendering on the CPU (no GPU found)")


# -------------------------------------------------------------------- main

ASPECTS = [(4, 3), (3, 4), (16, 9), (9, 16), (1, 1), (3, 2)]


def done_so_far(labels_path):
    if not os.path.exists(labels_path):
        return 0
    with open(labels_path) as f:
        return sum(1 for line in f if line.strip())


def main(argv):
    p = argparse.ArgumentParser(description="Render labelled chess board photos")
    p.add_argument("--positions", required=True, help="FEN placements, one per line (positions.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=100, help="photos to have in --out when done")
    p.add_argument("--views", type=int, default=3, help="photos of each position (different scenes and angles)")
    p.add_argument("--size", type=int, default=1024, help="long side in pixels")
    p.add_argument("--samples", type=int, default=32, help="Cycles samples (denoised)")
    p.add_argument("--style-every", type=int, default=25, help="a new piece design every this many photos")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="don't look for a GPU")
    a = p.parse_args(argv)

    with open(a.positions) as f:
        positions = [line.strip() for line in f if line.strip()]
    for fen in positions[:50]:
        parse_placement(fen)
    os.makedirs(os.path.join(a.out, "photos"), exist_ok=True)
    labels_path = os.path.join(a.out, "labels.jsonl")
    start = done_so_far(labels_path)
    if start >= a.count:
        print(f"{a.out} already has {start} photos")
        return

    # Start from an empty scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = random.Random(a.seed * 1_000_003 + start)
    sc = Scene(rng, a)
    t0 = time.time()
    with open(labels_path, "a") as labels:
        for n in range(start, a.count):
            pos_index = n // max(a.views, 1)
            placement = positions[pos_index % len(positions)].split()[0]
            sc.build(placement)
            w, h = rng.choice(ASPECTS)
            long = a.size
            width, height = (long, round(long * h / w)) if w >= h else (round(long * w / h), long)
            if not sc.place_camera(width, height):
                print(f"photo {n}: no view found, skipped")
                continue
            name = f"photos/{n:06d}.jpg"
            sc.scene.render.filepath = os.path.join(a.out, name)
            bpy.ops.render.render(write_still=True)
            labels.write(json.dumps({"image": name, "placement": placement, "group": f"pos-{pos_index:06d}"}) + "\n")
            labels.flush()
            done = n + 1 - start
            if done == 1:
                # The first photo also compiles the GPU's shaders: leave it
                # out of the estimate.
                t1 = time.time()
                print(f"photo {n + 1}/{a.count} (the first one is slow: setting up)")
                continue
            rate = (time.time() - t1) / (done - 1)
            print(f"photo {n + 1}/{a.count} ({rate:.1f}s each, about {rate * (a.count - n - 1) / 60:.0f} min left)")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    main(argv)
