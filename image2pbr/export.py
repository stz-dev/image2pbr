"""Saving maps, ZIP bundles and the Blender material-setup script."""
from __future__ import annotations

import json
import os
import zipfile

import cv2
import numpy as np

from . import pbr

BLENDER_TEMPLATE = '''"""Image2PBR - creates a Blender material from the maps in this folder.

How to use: open Blender -> Scripting tab -> Open this file -> Run Script (Alt+P).
If a mesh object is active, the material is assigned to it.
"""
import bpy
import os

MATERIAL_NAME = "Image2PBR"
TILING = 1.0            # texture repeat (use a seamless base colour for tiling > 1)
DISPLACEMENT = 0.05     # height scale in metres (only if a height map exists)
TEXTURE_DIR = ""        # leave empty = the folder of this script

FILES = __FILES__       # role -> file name


def _dir():
    if TEXTURE_DIR:
        return bpy.path.abspath(TEXTURE_DIR)
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    try:
        text = bpy.context.space_data.text
        if text and text.filepath:
            return os.path.dirname(bpy.path.abspath(text.filepath))
    except AttributeError:
        pass
    raise RuntimeError("Set TEXTURE_DIR at the top of the script to the folder with the maps.")


DIR = _dir()
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


def tex(role, colorspace):
    img = bpy.data.images.load(os.path.join(DIR, FILES[role]), check_existing=True)
    img.colorspace_settings.name = colorspace
    node = nodes.new("ShaderNodeTexImage")
    node.image = img
    node.label = role
    node.location = (-300, 500 - 300 * _row[0])
    _row[0] += 1
    links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    return node


if "basecolor" in FILES:
    base = tex("basecolor", "sRGB")
    if "ao" in FILES:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.location = (300, 300)
        links.new(base.outputs["Color"], mix.inputs["Color1"])
        links.new(tex("ao", "Non-Color").outputs["Color"], mix.inputs["Color2"])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        links.new(base.outputs["Color"], bsdf.inputs["Base Color"])

if "roughness" in FILES:
    links.new(tex("roughness", "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
if "metallic" in FILES:
    links.new(tex("metallic", "Non-Color").outputs["Color"], bsdf.inputs["Metallic"])
if "normal" in FILES:
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.location = (300, -300)
    links.new(tex("normal", "Non-Color").outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
if "height" in FILES:
    disp = nodes.new("ShaderNodeDisplacement")
    disp.location = (600, -400)
    disp.inputs["Scale"].default_value = DISPLACEMENT
    disp.inputs["Midlevel"].default_value = 0.5
    links.new(tex("height", "Non-Color").outputs["Color"], disp.inputs["Height"])
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
