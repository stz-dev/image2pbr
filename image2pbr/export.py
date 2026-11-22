"""Saving maps, ZIP bundles and the Blender material-setup script."""
from __future__ import annotations

import json
import os
import zipfile

import cv2
import numpy as np

from . import pbr

BLENDER_TEMPLATE = r'''"""Image2PBR - creates a Blender material from the maps in this folder.

How to use: open Blender -> Scripting tab -> Open this file -> Run Script (Alt+P).
If a mesh object is active, the material is assigned to it.
"""
import bpy
import os

MATERIAL_NAME = "Image2PBR"
TILING = 1.0            # texture repeat (use a seamless base colour for tiling > 1)
DISPLACEMENT = 0.05     # height scale in metres (only if a height map exists)
TEXTURE_DIR = ""        # folder with the maps. Empty = search automatically (see below)

FILES = __FILES__       # role -> file name


def _candidate_dirs():
    """Folders where the maps may be: TEXTURE_DIR, next to this script, next to the .blend, cwd."""
    dirs = []
    if TEXTURE_DIR:
        dirs.append(bpy.path.abspath(TEXTURE_DIR))
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    try:
        text = bpy.context.space_data.text
        if text is not None and text.filepath:
            dirs.append(os.path.dirname(bpy.path.abspath(text.filepath)))
    except AttributeError:
        pass
    for t in bpy.data.texts:                       # any script opened in Blender's text editor
        if t.filepath and os.path.basename(bpy.path.abspath(t.filepath)) == "blender_setup.py":
            dirs.append(os.path.dirname(bpy.path.abspath(t.filepath)))
    if bpy.data.filepath:
        dirs.append(os.path.dirname(bpy.data.filepath))
    dirs.append(os.getcwd())
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _find_dir():
    tried = _candidate_dirs()
    for d in tried:
        if any(os.path.isfile(os.path.join(d, f)) for f in FILES.values()):
            return d
    raise RuntimeError(
        "Image2PBR: the texture files were not found.\n"
        "Expected files: %s\n"
        "Searched folders:\n  %s\n"
        "Fix: extract the whole ZIP first and keep blender_setup.py in the same folder as the PNG maps, "
        "or set TEXTURE_DIR at the top of this script, e.g. TEXTURE_DIR = r\"D:\\Textures\\wood\"."
        % (", ".join(FILES.values()), "\n  ".join(tried)))


DIR = _find_dir()
print("Image2PBR: using textures from", DIR)
mat = bpy.data.materials.get(MATERIAL_NAME) or bpy.data.materials.new(MATERIAL_NAME)
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
nodes.clear()

out = nodes.new("ShaderNodeOutputMaterial")
out.location = (900, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
bsdf.location = (600, 0)
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

coord = nodes.new("ShaderNodeTexCoord")
coord.location = (-900, 0)
mapping = nodes.new("ShaderNodeMapping")
mapping.location = (-700, 0)
mapping.inputs["Scale"].default_value = (TILING, TILING, TILING)
links.new(coord.outputs["UV"], mapping.inputs["Vector"])

_row = [0]


class _Missing:
    """Placeholder returned when a map file is missing: linking to it is silently skipped."""
    outputs = {"Color": None}


def tex(role, colorspace):
    path = os.path.join(DIR, FILES[role])
    if not os.path.isfile(path):
        print("Image2PBR: WARNING - file not found, skipped:", path)
        return _Missing()
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    node = nodes.new("ShaderNodeTexImage")
    node.image = img
    node.label = role
    node.location = (-300, 500 - 300 * _row[0])
    _row[0] += 1
    links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    return node


def connect(node, socket):
    """Link a texture node's colour to a socket unless the texture was missing."""
    if not isinstance(node, _Missing):
        links.new(node.outputs["Color"], socket)


if "basecolor" in FILES:
    base = tex("basecolor", "sRGB")
    ao_node = tex("ao", "Non-Color") if "ao" in FILES else _Missing()
    if not isinstance(ao_node, _Missing) and not isinstance(base, _Missing):
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.location = (300, 300)
        connect(base, mix.inputs["Color1"])
        connect(ao_node, mix.inputs["Color2"])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        connect(base, bsdf.inputs["Base Color"])

if "roughness" in FILES:
    connect(tex("roughness", "Non-Color"), bsdf.inputs["Roughness"])
if "metallic" in FILES:
    connect(tex("metallic", "Non-Color"), bsdf.inputs["Metallic"])
if "normal" in FILES:
    normal_tex = tex("normal", "Non-Color")
    if not isinstance(normal_tex, _Missing):
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (300, -300)
        connect(normal_tex, nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
if "height" in FILES:
    height_tex = tex("height", "Non-Color")
    if not isinstance(height_tex, _Missing):
        disp = nodes.new("ShaderNodeDisplacement")
        disp.location = (600, -400)
        disp.inputs["Scale"].default_value = DISPLACEMENT
        disp.inputs["Midlevel"].default_value = 0.5
        connect(height_tex, disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
        try:
            mat.displacement_method = "BOTH"            # Blender 4.2+
        except Exception:
            try:
                mat.cycles.displacement_method = "BOTH"  # older versions
            except Exception:
                pass

obj = bpy.context.active_object
if obj is not None and obj.type == "MESH":
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
print("Image2PBR: material '%s' created from %s" % (MATERIAL_NAME, DIR))
'''

README_TXT = """Image2PBR export
================
basecolor.png  - source colour (sRGB)
roughness.png / metallic.png / ao.png - greyscale, import as Non-Color
normal.png     - OpenGL normal map (Blender), import as Non-Color
height.png     - 16-bit displacement, import as Non-Color
params.json    - exact parameters used (reproducible)
blender_setup.py - Blender -> Scripting -> Open -> Run Script: builds the material for you
"""


def blender_script(files: dict) -> str:
    return BLENDER_TEMPLATE.replace("__FILES__", repr(files))


def png_bytes(arr: np.ndarray) -> bytes:
    """Encode uint8/uint16 gray or RGB array as PNG (unicode-path safe, supports 16-bit)."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def save_png(path: str, arr: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(png_bytes(arr))
    return path


def write_bundle(zf: zipfile.ZipFile, prefix: str, rgb: np.ndarray, pal: dict | None,
                 params_by_map: dict, max_side: int | None) -> list[str]:
    """Render every map in params_by_map at full size and write them (plus Blender script) into zf."""
    base = pbr.fit(rgb, max_side)
    files = {"basecolor": "basecolor.png"}
    zf.writestr(prefix + "basecolor.png", png_bytes(base))
    for name, params in params_by_map.items():
        arr = pbr.render(rgb, name, params, pal, max_side=max_side, bit16=(name == "height"))
        files[name] = f"{name}.png"
        zf.writestr(prefix + files[name], png_bytes(arr))
    zf.writestr(prefix + "blender_setup.py", blender_script(files))
    zf.writestr(prefix + "params.json", json.dumps(params_by_map, indent=2, ensure_ascii=False))
    zf.writestr(prefix + "README.txt", README_TXT)
    return list(files)
