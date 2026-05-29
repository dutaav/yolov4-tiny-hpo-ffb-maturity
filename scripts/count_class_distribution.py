from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("palm-class-distribution")

volume = modal.Volume.from_name("palm-yolov4-vol")
VOL = "/vol"
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOL: volume}, timeout=3600)
def count() -> dict:
    import subprocess

    info = json.loads(Path(f"{VOL}/dataset_info.json").read_text())
    class_names = info["class_names"]
    base = Path(f"{VOL}/dataset/palm-ripeness-detection-5")

    result: dict = {"class_names": class_names, "splits": {}}
    for split in ["train", "valid", "test"]:
        split_dir = base / split
        counts = [0] * len(class_names)

        images = [
            p for p in split_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        n_images = len(images)

        script = (
            f'find "{split_dir}" -maxdepth 1 -name "*.txt" '
            '! -name "_darknet.labels" -exec awk "{print \\$1}" {} + 2>/dev/null '
            "| sort -n | uniq -c"
        )
        proc = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True
        )
        n_boxes = 0
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            cnt, cid = int(parts[0]), int(parts[1])
            if 0 <= cid < len(counts):
                counts[cid] += cnt
                n_boxes += cnt

        result["splits"][split] = {
            "n_images": n_images,
            "n_boxes": n_boxes,
            "per_class": dict(zip(class_names, counts)),
        }
        print(f"[{split}] images={n_images} boxes={n_boxes}")
    return result


@app.local_entrypoint()
def main() -> None:
    data = count.remote()
    out = (
        Path(__file__).resolve().parent.parent
        / "backup"
        / "paper_handoff"
        / "data"
        / "class_distribution.json"
    )
    out.write_text(json.dumps(data, indent=2))

    names = data["class_names"]
    print(f"\n{'class':<18}" + "".join(f"{s:>10}" for s in ["train", "valid", "test"]))
    print("-" * 48)
    totals = {s: 0 for s in ["train", "valid", "test"]}
    for i, name in enumerate(names):
        row = f"{name:<18}"
        for s in ["train", "valid", "test"]:
            c = data["splits"][s]["per_class"][name]
            totals[s] += c
            row += f"{c:>10}"
        print(row)
    print("-" * 48)
    print(f"{'total boxes':<18}" + "".join(f"{totals[s]:>10}" for s in ["train", "valid", "test"]))
    print(f"{'total images':<18}" + "".join(f"{data['splits'][s]['n_images']:>10}" for s in ["train", "valid", "test"]))
    print(f"\nsaved: {out}")
