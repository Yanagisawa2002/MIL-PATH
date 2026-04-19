"""Export KG graph edges and split supervision tables for external TxGNN runs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, write_csv

CLINICAL_RELATIONS = {"indication", "contraindication", "off-label use"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/experiments/full_fast_random.yaml",
            "configs/experiments/full_fast_cold_drug.yaml",
            "configs/experiments/full_fast_cold_disease.yaml",
        ],
    )
    parser.add_argument("--output-dir", type=str, default="exports/txgnn_bundle")
    return parser.parse_args()


def _should_keep_edge(row: dict[str, str], keep_only_types: set[str]) -> bool:
    return row["x_type"] in keep_only_types and row["y_type"] in keep_only_types


def _is_direct_clinical_pair(row: dict[str, str]) -> bool:
    pair_types = {row["x_type"], row["y_type"]}
    return pair_types == {"drug", "disease"} and row["display_relation"] in CLINICAL_RELATIONS


def _safe_metadata_repr(raw: str) -> dict[str, Any]:
    if not raw or raw == "{}":
        return {}
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"raw_metadata": raw}


def _simplify_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for row in rows:
        metadata = _safe_metadata_repr(row.get("metadata", ""))
        simplified.append(
            {
                "pair_id": row["pair_id"],
                "drug_id": row["drug_id"],
                "disease_id": row["disease_id"],
                "label": int(row["label"]),
                "split": row["split"],
                "pair_source": row["pair_source"],
                "has_gold_rationale": row["has_gold_rationale"] == "True",
                "num_gold_paths": int(row["num_gold_paths"]),
                "matched_positive_pair": metadata.get("matched_positive_pair", ""),
            }
        )
    return simplified


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_json(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _export_split_tables(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    project_name = config["project"]["name"]
    split_dir = Path(config["paths"]["split_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    split_output_dir = ensure_dir(output_root / project_name)

    summary: dict[str, Any] = {"project_name": project_name, "splits": {}}
    for split_name in ("train", "valid", "test"):
        src_path = split_dir / f"pairs_{split_name}.csv"
        rows = _load_csv_rows(src_path)
        simplified = _simplify_pair_rows(rows)
        positives = [row for row in simplified if int(row["label"]) == 1]
        negatives = [row for row in simplified if int(row["label"]) == 0]

        write_csv(rows, split_output_dir / f"pairs_{split_name}.csv")
        write_csv(simplified, split_output_dir / f"pairs_{split_name}_simple.csv")
        write_csv(positives, split_output_dir / f"pairs_{split_name}_positive.csv")
        write_csv(negatives, split_output_dir / f"pairs_{split_name}_negative.csv")

        summary["splits"][split_name] = {
            "num_pairs": len(simplified),
            "num_positive": len(positives),
            "num_negative": len(negatives),
            "num_gold_rationale_pairs": sum(int(row["has_gold_rationale"]) for row in simplified),
        }

    _copy_json(processed_dir / "split_audit.json", split_output_dir / "split_audit.json")
    (split_output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {project_name}",
                "",
                "Files:",
                "- `pairs_train.csv` / `pairs_valid.csv` / `pairs_test.csv`: original split rows from this project.",
                "- `pairs_*_simple.csv`: simplified pair supervision with parsed metadata.",
                "- `pairs_*_positive.csv` / `pairs_*_negative.csv`: positive and negative subsets.",
                "- `split_audit.json`: split leakage audit summary.",
                "",
                "All splits share the common graph under `../common_graph/`.",
            ]
        ),
        encoding="utf-8",
    )
    save_json(summary, split_output_dir / "summary.json")
    return summary


def _export_common_graph(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    processed_dir = Path(config["paths"]["processed_dir"])
    raw_kg_path = Path(config["paths"]["raw_kg_csv"])
    build_cfg = config["data"]["build"]
    keep_only_types = set(build_cfg["keep_only_types"])
    remove_direct = bool(build_cfg["remove_direct_drug_disease_edges_from_graph"])

    common_dir = ensure_dir(output_root / "common_graph")
    graph_data = __import__("torch").load(processed_dir / "graph_data.pt", weights_only=False)
    graph_node_to_idx = graph_data["node_to_idx"]
    relation_rows = [
        {"relation_id": idx, "relation": relation}
        for idx, relation in enumerate(graph_data["relation_by_idx"])
    ]
    node_rows: dict[str, dict[str, Any]] = {}
    kept_edges: list[dict[str, str]] = []
    clinical_rows: list[dict[str, str]] = []

    with raw_kg_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _should_keep_edge(row, keep_only_types):
                continue
            is_clinical = _is_direct_clinical_pair(row)
            if is_clinical:
                clinical_rows.append(row)
                if remove_direct:
                    continue
            kept_edges.append(row)
            for prefix in ("x", "y"):
                node_key = row[f"{prefix}_node_key"] or f"{row[f'{prefix}_type']}:{row[f'{prefix}_id']}"
                if node_key not in graph_node_to_idx:
                    continue
                node_rows.setdefault(
                    node_key,
                    {
                        "node_key": node_key,
                        "node_index": int(graph_node_to_idx[node_key]),
                        "node_id": row[f"{prefix}_id"],
                        "node_type": row[f"{prefix}_type"],
                        "node_name": row[f"{prefix}_name"],
                        "node_source": row[f"{prefix}_source"],
                    },
                )

    write_csv(kept_edges, common_dir / "graph_edges.csv")
    write_csv(clinical_rows, common_dir / "clinical_drug_disease_edges.csv")
    write_csv(sorted(node_rows.values(), key=lambda item: item["node_index"]), common_dir / "node_catalog.csv")
    write_csv(relation_rows, common_dir / "relation_vocab.csv")
    save_json(
        {
            "num_graph_nodes": int(graph_data["metadata"]["num_nodes"]),
            "num_graph_edges": len(kept_edges),
            "num_direct_clinical_rows": len(clinical_rows),
            "remove_direct_drug_disease_edges_from_graph": remove_direct,
            "keep_only_types": sorted(keep_only_types),
        },
        common_dir / "summary.json",
    )
    (common_dir / "README.md").write_text(
        "\n".join(
            [
                "# Common Graph",
                "",
                "- `graph_edges.csv`: KG edges used in our graph construction after filtering node types and removing direct clinical drug-disease edges.",
                "- `clinical_drug_disease_edges.csv`: all direct clinical drug-disease rows from the raw KG.",
                "- `node_catalog.csv`: node metadata for graph nodes.",
                "- `relation_vocab.csv`: relation id to relation name mapping from `graph_data.pt`.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "num_graph_nodes": int(graph_data["metadata"]["num_nodes"]),
        "num_graph_edges": len(kept_edges),
        "num_direct_clinical_rows": len(clinical_rows),
    }


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_dir))
    prepared_configs = [
        prepare_experiment_config(load_experiment_config(config_path), repo_root=REPO_ROOT)
        for config_path in args.configs
    ]

    common_summary = _export_common_graph(prepared_configs[0], output_root)
    split_summaries = [
        _export_split_tables(config, output_root)
        for config in prepared_configs
    ]

    save_json(
        {
            "common_graph": common_summary,
            "projects": split_summaries,
        },
        output_root / "manifest.json",
    )
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# TxGNN Export Bundle",
                "",
                "This directory contains:",
                "- `common_graph/`: graph edges and vocab shared by all exported splits.",
                "- `<project_name>/`: split-specific train/valid/test pair supervision tables.",
                "",
                "Exported projects:",
                *[f"- `{summary['project_name']}`" for summary in split_summaries],
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
