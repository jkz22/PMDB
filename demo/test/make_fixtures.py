import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp
from scripts.demo_confound import FEATURES as CONFOUND_FEATURES
from scripts.demo_confound import load_bse_stats, nearest_centroid


def records_from_frame(frame):
    return [
        {
            "batch": str(batch),
            "site": str(site),
            **{str(column): float(value) for column, value in row.items()},
        }
        for (batch, site), row in frame.iterrows()
    ]


def predictions_from_frame(frame):
    predictions = []
    for (batch, site), row in frame.iterrows():
        predictions.append(
            {
                "batch": str(batch),
                "site": str(site),
                "assigned": str(row["assigned"]),
                "credibility": float(row["credibility"]),
                "confidence": float(row["confidence"]),
                "ood": bool(row["ood"]),
                "p": {
                    batch_name: float(row[f"p_{batch_name}"])
                    for batch_name in ("Batch_1", "Batch_2", "Batch_3")
                },
                "score": {
                    batch_name: float(row[f"score_{batch_name}"])
                    for batch_name in ("Batch_1", "Batch_2", "Batch_3")
                },
            }
        )
    return predictions


def prediction_matches_csv(predictions, expected):
    expected_by_site = {
        (str(row["batch"]), str(row["site"])): row for _, row in expected.iterrows()
    }
    for prediction in predictions:
        row = expected_by_site[(prediction["batch"], prediction["site"])]
        assert prediction["assigned"] == row["assigned"]
        assert prediction["ood"] == bool(row["ood"])
        for field in ("credibility", "confidence"):
            assert abs(prediction[field] - float(row[field])) <= 1e-12
        for batch in ("Batch_1", "Batch_2", "Batch_3"):
            assert abs(prediction["p"][batch] - float(row[f"p_{batch}"])) <= 1e-12
            assert abs(prediction["score"][batch] - float(row[f"score_{batch}"])) <= 1e-12


def main():
    feature_names = json.loads(
        (ROOT / "outputs/fingerprint/evaluation.json").read_text()
    )["features"]
    fingerprint_inputs = fp.read_feature_inputs(
        ROOT / "outputs/kpis/curves.csv", ROOT / "outputs/kpis/tile_kpis.csv"
    )
    expected_features = pd.read_csv(
        ROOT / "outputs/fingerprint/features.csv",
        dtype={"batch": str, "site": str},
    ).set_index(["batch", "site"])[feature_names]
    fingerprint_inputs = fingerprint_inputs[feature_names]
    assert fingerprint_inputs.index.equals(expected_features.index)
    feature_diff = np.abs(
        fingerprint_inputs.to_numpy(dtype=float) - expected_features.to_numpy(dtype=float)
    )
    max_feature_diff = float(feature_diff.max()) if feature_diff.size else 0.0
    features_match = bool(np.all(feature_diff <= 1e-12))
    print(
        f"features.csv matches read_feature_inputs to 1e-12: {features_match} "
        f"(max abs diff {max_feature_diff:.17g})"
    )
    assert features_match, "read_feature_inputs differs from outputs/fingerprint/features.csv"

    labels = fingerprint_inputs.index.get_level_values("batch").astype(str)
    model = fp.fit(fingerprint_inputs, pd.Series(labels, index=fingerprint_inputs.index))
    model_fixture = {
        "features": model.features,
        "batches": model.batches,
        "center": model.center.to_list(),
        "scale": model.scale.to_list(),
        "batchCenter": model.batch_center.to_numpy().tolist(),
        "batchScale": model.batch_scale.to_numpy().tolist(),
        "trainX": model.train_x.tolist(),
        "trainCodes": model.train_codes.tolist(),
    }

    heldout_features = pd.read_csv(
        ROOT / "outputs/fingerprint/heldout_features.csv",
        dtype={"batch": str, "site": str},
    ).set_index(["batch", "site"])[feature_names]
    heldout_predictions = fp.predict(model, heldout_features)
    heldout_reference = predictions_from_frame(heldout_predictions)
    expected_predictions_frame = pd.read_csv(
        ROOT / "outputs/fingerprint/heldout_predictions.csv",
        dtype={"batch": str, "site": str, "assigned": str},
    ).set_index(["batch", "site"])
    prediction_matches_csv(heldout_reference, expected_predictions_frame.reset_index())

    batch3_p = fp.conformal_p(model, fingerprint_inputs)["Batch_3"]
    is_batch3 = fingerprint_inputs.index.get_level_values("batch") == "Batch_3"
    base_index = batch3_p[is_batch3].idxmax()
    base = fingerprint_inputs.loc[base_index]
    drift_grid = []
    for step in range(41):
        t = step * 0.05
        drifted = pd.DataFrame(
            [fp_drift(base, t)],
            index=pd.MultiIndex.from_tuples(
                [("Batch_N", f"drift_{t:.2f}")], names=["batch", "site"]
            ),
        )[model.features]
        pred = fp.predict(model, drifted).iloc[0]
        drift_grid.append(
            {
                "t": t,
                "features": {
                    feature: float(drifted.iloc[0][feature])
                    for feature in model.features
                },
                "prediction": {
                    "assigned": str(pred["assigned"]),
                    "credibility": float(pred["credibility"]),
                    "confidence": float(pred["confidence"]),
                    "ood": bool(pred["ood"]),
                    "p": {
                        batch: float(pred[f"p_{batch}"])
                        for batch in model.batches
                    },
                    "score": {
                        batch: float(pred[f"score_{batch}"])
                        for batch in model.batches
                    },
                },
            }
        )

    x = fingerprint_inputs[model.features].to_numpy(dtype=float)
    codes = np.asarray([model.batches.index(batch) for batch in labels])
    assignments = fp._loo_assignments(x, codes, len(model.batches))
    loo_assignments = [
        {
            "batch": str(batch),
            "site": str(site),
            "assigned": model.batches[int(assignment)],
        }
        for (batch, site), assignment in zip(fingerprint_inputs.index, assignments)
    ]
    loo_correct = int(
        sum(row["batch"] == row["assigned"] for row in loo_assignments)
    )
    assert loo_correct == 21 and len(loo_assignments) == 31

    confound_frame = load_bse_stats()
    confound_rows = confound_frame[CONFOUND_FEATURES]
    confound_labels = confound_frame.index.get_level_values("batch").astype(str).to_list()
    confound_accuracy = 0
    for i, row in enumerate(confound_rows.to_numpy(dtype=float)):
        keep = np.arange(len(confound_rows)) != i
        predicted = nearest_centroid(
            confound_rows.iloc[keep],
            np.asarray(confound_labels)[keep],
            row,
        )
        confound_accuracy += predicted == confound_labels[i]
    confound_loo_accuracy = confound_accuracy / len(confound_rows)

    confound_sweeps = []
    all_conf_rows = confound_rows.to_numpy(dtype=float)
    all_conf_labels = np.asarray(confound_labels)
    for i, ((batch, site), source) in enumerate(
        zip(confound_rows.index, all_conf_rows)
    ):
        if batch not in ("Batch_1", "Batch_2"):
            continue
        sweep = []
        for delta in range(26):
            query = np.asarray(
                [
                    source[0] + delta,
                    source[1] + delta,
                    0.0 if delta > 0 else source[2],
                ],
                dtype=float,
            )
            sweep.append(
                {
                    "delta": delta,
                    "features": query.tolist(),
                    "assigned": nearest_centroid(
                        confound_rows, all_conf_labels, query
                    ),
                }
            )
        confound_sweeps.append(
            {"batch": str(batch), "site": str(site), "predictions": sweep}
        )
    assert round(confound_loo_accuracy, 2) == 0.71

    fixtures = {
        "featuresMatch": features_match,
        "maxFeatureDiff": max_feature_diff,
        "fingerprint": {
            "features": feature_names,
            "model": model_fixture,
            "inputs": {
                "trainRows": records_from_frame(fingerprint_inputs),
                "heldoutRows": records_from_frame(heldout_features),
            },
            "heldout": {
                "predictions": heldout_reference,
                "csvMatched": True,
            },
            "drift": {
                "base": {
                    "batch": str(base_index[0]),
                    "site": str(base_index[1]),
                    "features": {
                        feature: float(base[feature]) for feature in model.features
                    },
                },
                "grid": drift_grid,
            },
            "loo": {
                "assignments": loo_assignments,
                "correct": loo_correct,
                "n": len(loo_assignments),
                "accuracy": loo_correct / len(loo_assignments),
            },
        },
        "confound": {
            "features": CONFOUND_FEATURES,
            "rows": [
                {
                    "batch": str(batch),
                    "site": str(site),
                    **{
                        feature: float(value)
                        for feature, value in zip(
                            CONFOUND_FEATURES, row
                        )
                    },
                }
                for (batch, site), row in zip(confound_rows.index, all_conf_rows)
            ],
            "looAccuracy": confound_loo_accuracy,
            "sweeps": confound_sweeps,
        },
    }
    output_path = ROOT / "demo/test/fixtures.json"
    output_path.write_text(json.dumps(fixtures, indent=2) + "\n")
    print(f"Wrote {output_path.relative_to(ROOT)}")


def fp_drift(row, t):
    result = row.copy()
    result["si_depth_rel_band0"] *= 1 - 0.5 * t
    result["si_depth_rel_band1"] *= 1 - 0.3 * t
    result["si_depth_rel_band3"] *= 1 + 0.4 * t
    result["si_depth_rel_band4"] *= 1 + 0.8 * t
    result["si_depth_slope"] = (
        result["si_depth_rel_band4"] - result["si_depth_rel_band0"]
    )
    result["si_depth_mid_dip"] = result["si_depth_rel_band2"] - (
        result["si_depth_rel_band0"] + result["si_depth_rel_band4"]
    ) / 2
    for feature in result.index:
        if feature.startswith(("gx_", "gz_")):
            result[feature] *= 1 - 0.35 * t
    result["k15_contact_tilestd"] *= 1 + 4 * t
    return result


if __name__ == "__main__":
    main()
