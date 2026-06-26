"""Prepare tiny demo datasets for DB15K, MKG-W, and MKG-Y experiments.

The full benchmark packages are public research resources, but their visible
repository READMEs do not provide a clear redistribution license for mirroring
the raw files. This script therefore avoids full automatic downloads. For each
dataset, it tries to fetch a small public source README as provenance evidence,
then writes a tiny loader-compatible demo split with random text/image features.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


DATASET_SOURCES = {
    "DB15K": {
        "base": "DBpedia",
        "source_readmes": [
            "https://raw.githubusercontent.com/quqxui/MMRNS/main/README.md",
            "https://raw.githubusercontent.com/quqxui/MMRNS/master/README.md",
        ],
    },
    "MKG-W": {
        "base": "Wikidata",
        "source_readmes": [
            "https://raw.githubusercontent.com/ADMIS-TONGJI/C2RS/main/README.md",
            "https://raw.githubusercontent.com/ADMIS-TONGJI/C2RS/master/README.md",
        ],
    },
    "MKG-Y": {
        "base": "YAGO",
        "source_readmes": [
            "https://raw.githubusercontent.com/ngl567/DHNS/main/README.md",
            "https://raw.githubusercontent.com/ngl567/DHNS/master/README.md",
        ],
    },
}


RELATIONS = ["relatedTo", "locatedIn", "createdBy", "partOf"]


def fetch_small_text(url, timeout=20, max_bytes=65536):
    """Download at most max_bytes from a public text URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "HERO-demo-dataset-script/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def write_mapping(path, names):
    """Write a name-to-id mapping accepted by KGDataset."""
    with open(path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(names):
            f.write(f"{name}\t{idx}\n")


def write_triples(path, triples):
    """Write triples in text form so the loader exercises mapping logic."""
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def build_demo_triples(dataset_name, entity_names):
    """Create deterministic toy triples with dataset-specific entity prefixes."""
    triples = [
        (entity_names[0], RELATIONS[0], entity_names[1]),
        (entity_names[1], RELATIONS[1], entity_names[2]),
        (entity_names[2], RELATIONS[2], entity_names[3]),
        (entity_names[3], RELATIONS[3], entity_names[4]),
        (entity_names[4], RELATIONS[0], entity_names[5]),
        (entity_names[5], RELATIONS[1], entity_names[0]),
    ]
    # Keep one relation pattern unique per dataset so demo outputs are not
    # byte-identical across DB15K, MKG-W, and MKG-Y.
    triples.append((entity_names[0], RELATIONS[2], f"{dataset_name}_demo_anchor"))
    return triples


def prepare_one_dataset(output_root, dataset_name, text_dim, visual_dim, seed, skip_source_check):
    """Create one demo dataset directory and return a metadata dictionary."""
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    source = DATASET_SOURCES[dataset_name]
    source_status = {"ok": False, "url": None, "error": None}
    if not skip_source_check:
        errors = []
        for url in source["source_readmes"]:
            try:
                readme = fetch_small_text(url)
                (dataset_dir / "SOURCE_README_SNIPPET.txt").write_text(readme, encoding="utf-8")
                source_status.update({"ok": True, "url": url, "error": None})
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url}: {exc}")
        if not source_status["ok"]:
            source_status["error"] = " | ".join(errors)

    entity_names = [f"{dataset_name}_entity_{idx}" for idx in range(6)] + [f"{dataset_name}_demo_anchor"]
    triples = build_demo_triples(dataset_name, entity_names)
    train, valid, test = triples[:5], triples[5:6], triples[6:]

    write_mapping(dataset_dir / "entity2id.txt", entity_names)
    write_mapping(dataset_dir / "relation2id.txt", RELATIONS)
    write_triples(dataset_dir / "train.txt", train)
    write_triples(dataset_dir / "valid.txt", valid)
    write_triples(dataset_dir / "test.txt", test)

    rng = np.random.default_rng(seed + list(DATASET_SOURCES).index(dataset_name))
    np.save(dataset_dir / "text_features.npy", rng.normal(size=(len(entity_names), text_dim)).astype("float32"))
    np.save(dataset_dir / "visual_features.npy", rng.normal(size=(len(entity_names), visual_dim)).astype("float32"))

    metadata = {
        "dataset": dataset_name,
        "base_knowledge_source": source["base"],
        "demo_only": True,
        "entity_count": len(entity_names),
        "relation_count": len(RELATIONS),
        "train_count": len(train),
        "valid_count": len(valid),
        "test_count": len(test),
        "source_check": source_status,
        "note": "This is a tiny synthetic demo split, not the full public benchmark.",
    }
    (dataset_dir / "demo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    """Prepare all demo datasets from a single command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/demo", help="Directory that will contain DB15K, MKG-W, and MKG-Y demo folders.")
    parser.add_argument("--text-dim", type=int, default=16)
    parser.add_argument("--visual-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-source-check", action="store_true", help="Skip tiny public README downloads and only create local demo data.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    for dataset_name in DATASET_SOURCES:
        metadata = prepare_one_dataset(
            output_root,
            dataset_name,
            args.text_dim,
            args.visual_dim,
            args.seed,
            args.skip_source_check,
        )
        all_metadata.append(metadata)
        status = "downloaded source snippet" if metadata["source_check"]["ok"] else "source snippet unavailable"
        print(f"{dataset_name}: wrote demo data to {output_root / dataset_name} ({status})")

    (output_root / "demo_manifest.json").write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 