"""Parent-image table from committed CSVs (no data needed)."""

from __future__ import annotations

from pmdb.parents import parent_groups, parent_id

EXPECTED = {"h1612_ETD_s1": 2, "h1780_ETD_s1": 1, "h1880_ETD_s1": 1, "h1904_ETD_s1": 3,
            "h2048_ETD_s2": 2, "h2060_ETD_s1": 4, "h2068_SE_s1": 4, "h2080_ETD_s1": 3,
            "h2088_ETD_s3": 3, "h2148_ETD_s1": 2, "h2156_ETD_s2": 2, "h2272_ETD_s2": 2,
            "h2316_ETD_s2": 2}


def test_labelled_parents():
    df = parent_groups()
    lab = df[df["batch"] != "Batch_heldout"]
    assert len(lab) == 31 and lab["parent_id"].nunique() == 13
    assert lab["parent_id"].value_counts().to_dict() == EXPECTED


def test_heldout_parents():
    df = parent_groups().set_index("site")
    assert df.loc["3e122cbj", "parent_id"] == "h2316_ETD_s2"
    assert df.loc["fn0mhxef", "parent_id"] == "h2048_ETD_s2"
    assert df.loc["xrv9xvzb", "parent_id"] == "h2088_ETD_s3"


def test_batches_have_six_parents():
    lab = parent_groups()
    lab = lab[lab["batch"] != "Batch_heldout"]
    assert (lab.groupby("batch")["parent_id"].nunique() >= 6).all()


def test_parent_id():
    assert parent_id(2068, "SE", 1) == "h2068_SE_s1"
