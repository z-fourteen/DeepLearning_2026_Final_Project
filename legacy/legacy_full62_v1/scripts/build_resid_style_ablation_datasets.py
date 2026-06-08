from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_NPZ = PROJECT_ROOT / "data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026.npz"
BASE_CONFIG = PROJECT_ROOT / "configs/sequence_gru_l20_clean_alpha_resid_style_strictmask_leaky0005.yaml"
OUT_DIR = PROJECT_ROOT / "data/mart/datasets"
CONFIG_DIR = PROJECT_ROOT / "configs"
MANIFEST_PATH = PROJECT_ROOT / "outputs/analysis/resid_style_ablation_manifest.json"

RESIDUAL_FEATURES = [
    "lag1_turnover_cost_proxy__resid_style",
    "lag1_turnover_20d_std__resid_style",
    "lag1_turnover_60d_std__resid_style",
    "lag1_amount_rank_pct__resid_style",
    "lag1_amount_log__resid_style",
]

SLUGS = {
    "lag1_turnover_cost_proxy__resid_style": "drop_turnover_cost_proxy_resid",
    "lag1_turnover_20d_std__resid_style": "drop_turnover_20d_std_resid",
    "lag1_turnover_60d_std__resid_style": "drop_turnover_60d_std_resid",
    "lag1_amount_rank_pct__resid_style": "drop_amount_rank_pct_resid",
    "lag1_amount_log__resid_style": "drop_amount_log_resid",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    data = np.load(BASE_NPZ, allow_pickle=True)
    feature_names = [str(item) for item in data["feature_names"]]
    base_config = load_yaml(BASE_CONFIG)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for dropped_feature in RESIDUAL_FEATURES:
        if dropped_feature not in feature_names:
            raise ValueError(f"Missing residual feature in base NPZ: {dropped_feature}")

        slug = SLUGS[dropped_feature]
        keep_indices = [idx for idx, name in enumerate(feature_names) if name != dropped_feature]
        kept_features = [feature_names[idx] for idx in keep_indices]
        output_stem = f"dataset_seq_l20_adv_clean_v1_alpha_resid_style_{slug}_chinext_2016_2026"
        npz_path = OUT_DIR / f"{output_stem}.npz"

        np.savez_compressed(
            npz_path,
            X=data["X"][:, :, keep_indices].astype("float32"),
            y=data["y"].astype("float32"),
            trade_date=data["trade_date"],
            ts_code=data["ts_code"],
            split=data["split"],
            feature_names=np.asarray(kept_features),
            build_mode=np.asarray([f"alpha_plus_residual_style__{slug}"]),
            dropped_feature=np.asarray([dropped_feature]),
        )

        config = dict(base_config)
        config["run"] = dict(base_config["run"])
        config["data"] = dict(base_config["data"])
        config["model"] = dict(base_config["model"])
        config["run"]["name"] = f"gru_l20_alpha_resid_style_{slug}_leaky0005"
        config["run"]["output_dir"] = f"outputs/runs/{config['run']['name']}"
        config["data"]["npz_path"] = str(npz_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        config["model"]["num_features"] = len(kept_features)

        config_path = CONFIG_DIR / f"sequence_gru_l20_alpha_resid_style_{slug}_leaky0005.yaml"
        write_yaml(config_path, config)

        results.append(
            {
                "dropped_feature": dropped_feature,
                "slug": slug,
                "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "num_features": len(kept_features),
                "samples": int(len(data["y"])),
            }
        )

    MANIFEST_PATH.write_text(json.dumps({"ablations": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ablations": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
