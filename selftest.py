"""Self-check: keys, network, and a full Qwen -> Gemini -> maps run on one image.

    python selftest.py                 # procedural wood texture
    python selftest.py photo.jpg       # your own image

Output: selftest_out/ (source, all maps, sheet.png, blender_setup.py).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

from image2pbr import ai, export, pbr, samples, settings

OUT = "selftest_out"


def main():
    settings.load_env()
    os.makedirs(OUT, exist_ok=True)

    print("== API keys")
    ok = True
    for svc in ("gemini", "groq"):
        r = ai.check_service(svc)
        name = ai.SERVICE_NAMES[svc]
        if r["missing"]:
            print(f"  {name}: NO KEY -> create one: {settings.KEY_LINKS[svc]}")
            ok = False
        elif r["ok"]:
            print(f"  {name}: OK ({len(r['models'])} models)")
        else:
            print(f"  {name}: FAILED - {r['message']}")
            ok = False
    if not ok:
        print("\nAI is unavailable, continuing with default parameters.")

    rgb = np.array(Image.open(sys.argv[1]).convert("RGB")) if len(sys.argv) > 1 else samples.make_wood()
    rgb = pbr.fit(rgb, 2048)
    Image.fromarray(rgb).save(f"{OUT}/source.png")
    pal = pbr.extract_palette(rgb, 6)
    print("\n== Palette:", [(p["hex"], p["share_percent"]) for p in pbr.palette_info(pal)])

    desc = "Generic material texture."
    if ok:
        print("\n== Qwen describes the texture...")
        try:
            desc = ai.describe_with_qwen(rgb, pal)
            print(desc)
        except ai.AIError as e:
            print("  Qwen failed:", e)

    results = {}
    if ok:
        print("\n== Gemini (one request for all maps)...")
        try:
            results = ai.params_for_maps(pbr.MAPS, desc, pal)
        except ai.AIError as e:
            print("  Gemini failed:", e)

    tiles = [rgb]
    for m in pbr.MAPS:
        params, notes = results.get(m, (pbr.defaults(m), "(defaults)"))
        print(f"  {m}: {notes}")
        out = pbr.render(rgb, m, params, pal, max_side=2048)
        export.save_png(f"{OUT}/{m}.png", out)
        out8 = (out >> 8).astype(np.uint8) if out.dtype == np.uint16 else out
        tiles.append(np.stack([out8] * 3, -1) if out8.ndim == 2 else out8)

    sheet = np.concatenate([cv2.resize(t, (400, 400), interpolation=cv2.INTER_AREA) for t in tiles], axis=1)
    Image.fromarray(sheet).save(f"{OUT}/sheet.png")
    with open(f"{OUT}/blender_setup.py", "w", encoding="utf-8") as fh:
        fh.write(export.blender_script({"basecolor": "source.png", **{m: f"{m}.png" for m in pbr.MAPS}}))
    print(f"\nDone: {OUT}/sheet.png  (source | roughness | metallic | normal | height | ao)")


if __name__ == "__main__":
    main()
