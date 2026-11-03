#!/usr/bin/env python
"""Resumable downloader for chlange Raman datasets on HuggingFace.

Usage:  python -I fetch_hf.py
Retries each file up to N times; skips files already complete.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE = "https://huggingface.co"
AUTHOR = "chlange"
DEST = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
DEST = r"D:\TIM\data\raw"

DATASETS = [
    "RamanSpectraBioprocessAnalystesAnton532",
    "RamanSpectraBioprocessAnalystesAnton785",
    "RamanSpectraBioprocessAnalystesKaiser",
    "RamanSpectraBioprocessAnalystesMetrohm",
    "RamanSpectraBioprocessAnalystesMettlerToledo",
    "RamanSpectraBioprocessAnalystesTec5",
    "RamanSpectraBioprocessAnalystesTimegate",
    "RamanSpectraBioprocessAnalystesTornado",
    "RamanSpectraEcoliFermentation",
    "SubstrateMixRaman",
    "HTRamanBioCatalysisAXP",
    "HTRamanBioCatalysisAXPlowSNR",
    "FuelRamanSpectraBenchtop",
]


def _get_json(url, tries=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 2 * i)
    raise last


def fetch(url, path, tries=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            return total
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  retry {i+1} {os.path.basename(path)}: {type(e).__name__}", flush=True)
            time.sleep(3 + 3 * i)
    raise last


def main():
    for ds in DATASETS:
        tree = _get_json(f"{BASE}/api/datasets/{AUTHOR}/{ds}/tree/main")
        print(f"== {ds}", flush=True)
        for s in tree:
            if s["type"] != "file" or s["path"].startswith("."):
                continue
            out = os.path.join(DEST, ds, s["path"])
            want = s.get("size", 0)
            if os.path.exists(out) and os.path.getsize(out) == want:
                print(f"  skip {s['path']} ({want/1e6:.2f}MB)", flush=True)
                continue
            url = f"{BASE}/datasets/{AUTHOR}/{ds}/resolve/main/{s['path']}"
            try:
                n = fetch(url, out)
                ok = "ok" if n == want else f"SIZE MISMATCH got={n} want={want}"
                print(f"  {ok} {s['path']} {n/1e6:.2f}MB", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {s['path']}: {type(e).__name__}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
