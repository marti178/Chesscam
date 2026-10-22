"""Baja recursos CC0 de Poly Haven (https://polyhaven.com, licencia CC0, API pública sin cuenta).

HDRI de interiores en 1k, el modelo «Chess Set» (reservado para test) y texturas de
madera y mármol para los tableros sintéticos. Descarga secuencial y reanudable,
y deja un `manifest.json` con la URL, el autor y la licencia de cada archivo.

  python -m chessscan.fetch_polyhaven --out data/raw/polyhaven
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
UA = {"User-Agent": "chess-live-scanner-ml/1.0 (uso personal, descarga secuencial)"}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dst)
    time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/polyhaven"))
    ap.add_argument("--res", default="1k")
    a = ap.parse_args()
    manifest = []
    hdris = get_json(f"{API}/assets?t=hdris&c=indoor")
    tex = {k: v for k, v in get_json(f"{API}/assets?t=textures").items()
           if any(c in ("wood", "marble") for c in v.get("categories", []))}
    jobs = [("hdri", k, v) for k, v in sorted(hdris.items())] + [("texture", k, v) for k, v in sorted(tex.items())]
    jobs.append(("model", "chess_set", get_json(f"{API}/info/chess_set")))
    for kind, key, info in jobs:
        try:
            files = get_json(f"{API}/files/{key}")
            if kind == "hdri":
                f = files["hdri"][a.res]["hdr"]
                download(f["url"], a.out / "hdri" / f"{key}.hdr")
            elif kind == "texture":
                for m in ("Diffuse", "Rough", "nor_gl"):
                    if m in files and a.res in files[m]:
                        ext = "jpg" if "jpg" in files[m][a.res] else next(iter(files[m][a.res]))
                        download(files[m][a.res][ext]["url"], a.out / "textures" / key / f"{m}.{ext}")
            else:
                g = files["gltf"][a.res]["gltf"]
                download(g["url"], a.out / "models" / key / f"{key}.gltf")
                for rel, inc in g.get("include", {}).items():
                    download(inc["url"], a.out / "models" / key / rel)
            manifest.append({"kind": kind, "id": key, "name": info.get("name"), "authors": info.get("authors"),
                             "license": "CC0 1.0", "source": f"https://polyhaven.com/a/{key}"})
            print(kind, key, flush=True)
        except Exception as e:  # noqa: BLE001 - seguir con el resto
            print("ERROR", kind, key, e, flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("TERMINADO", len(manifest))


if __name__ == "__main__":
    main()
