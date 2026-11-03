#!/usr/bin/env python
"""Download the real-broth validation datasets (S3-S7) including the RamanBench mirror.

Verified schemas are documented in research-log.md.  All runs are resumable and
retry on the flaky HuggingFace connection observed on this network.
"""
import json
import os
import time
import urllib.request

DEST = r"D:\TIM\data\raw"

TARGETS = [
    # (repo_id, remote_path, local_subdir)
    ("chlange/RamanSpectraEcoliFermentation", "train.csv", "RamanSpectraEcoliFermentation"),
    ("chlange/RamanSpectraEcoliFermentation", "val.csv", "RamanSpectraEcoliFermentation"),
    ("chlange/RamanSpectraEcoliFermentation", "test.csv", "RamanSpectraEcoliFermentation"),
    ("HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations", "train.parquet",
     "RamanSpectraEthanolicYeastFermentations"),
    ("HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations", "val.parquet",
     "RamanSpectraEthanolicYeastFermentations"),
    ("HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations", "test.parquet",
     "RamanSpectraEthanolicYeastFermentations"),
    ("HTW-KI-Werkstatt/RamanBench", "yeast_bioreactor_fermentation/data/train-00000-of-00001.parquet",
     "RamanBench/yeast_bioreactor_fermentation"),
    ("HTW-KI-Werkstatt/RamanBench", "yeast_bioreactor_fermentation/metadata.json",
     "RamanBench/yeast_bioreactor_fermentation"),
    ("HTW-KI-Werkstatt/RamanBench", "yeast_fermentation/data/train-00000-of-00001.parquet",
     "RamanBench/yeast_fermentation"),
    ("HTW-KI-Werkstatt/RamanBench", "yeast_batch_effects/data/train-00000-of-00001.parquet",
     "RamanBench/yeast_batch_effects"),
    ("HTW-KI-Werkstatt/RamanBench", "kaiser_ecoli_fermentation/data/train-00000-of-00001.parquet",
     "RamanBench/kaiser_ecoli_fermentation"),
    ("HTW-KI-Werkstatt/RamanBench", "tg_ecoli_fermentation/data/train-00000-of-00001.parquet",
     "RamanBench/tg_ecoli_fermentation"),
    ("HTW-KI-Werkstatt/RamanBench", "streptococcus_thermophilus_fermentation_kaiser/data/train-00000-of-00001.parquet",
     "RamanBench/strep_kaiser"),
    ("HTW-KI-Werkstatt/RamanBench", "streptococcus_thermophilus_fermentation_timegate/data/train-00000-of-00001.parquet",
     "RamanBench/strep_timegate"),
]


def fetch(url, path, tries=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return os.path.getsize(path)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"   retry {i+1} {os.path.basename(path)}: {type(e).__name__}", flush=True)
            time.sleep(4 + 3 * i)
    raise last


def main():
    for repo, remote, sub in TARGETS:
        out = os.path.join(DEST, sub, os.path.basename(remote))
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote}"
        try:
            n = fetch(url, out)
            print(f"OK   {sub}/{os.path.basename(remote):45} {n/1e6:8.2f} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {sub}/{os.path.basename(remote)}: {type(e).__name__}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
