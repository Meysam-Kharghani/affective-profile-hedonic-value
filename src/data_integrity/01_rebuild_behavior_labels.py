#!/usr/bin/env python3
"""
Build a conservative, EDF-independent behavioral label table for EmoEEG-MC.

Scope
-----
- Uses sub-01..sub-53 only (complete-trial participants with a published participant-specific sequence).
- Reconstructs stimulus_code and emotion_category from the participant-specific
  sequences documented in README.md.
- Does NOT guess imagery vs video block identity.
- Uses the published design: each consecutive pair of 3-trial blocks contains
  the same 3 stimulus codes once in imagery and once in video, with order randomized.

Thus, for each participant-specific 21-item sequence:
  seq[0:3], seq[0:3], seq[3:6], seq[3:6], ... seq[18:21], seq[18:21]
maps to the 42 behavioral rows.

This table is suitable for category/stimulus-dependent analyses that do not
require knowing which member of a paired block was imagery vs video.
"""

from __future__ import annotations
import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

RATING_NAMES = [
    "joy", "inspiration", "tenderness", "sadness", "fear", "disgust",
    "arousal", "valence", "familiarity", "liking",
]

PREFIX_TO_CATEGORY = {
    "joy": "joy",
    "ins": "inspiration",
    "ten": "tenderness",
    "sad": "sadness",
    "fear": "fear",
    "dis": "disgust",
    "neu": "neutral",
}

TARGET_COLUMN = {
    "joy": "joy",
    "inspiration": "inspiration",
    "tenderness": "tenderness",
    "sadness": "sadness",
    "fear": "fear",
    "disgust": "disgust",
}


def parse_full_subject_sequences(readme_path: Path):
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    mapping = {}

    for line in lines:
        m = re.match(r"^\s*(\d+)\.\s*`?(\[.*\])`?\s*$", line)
        if not m:
            continue
        sub = int(m.group(1))
        if not (1 <= sub <= 53):
            continue
        try:
            seq = ast.literal_eval(m.group(2))
        except Exception:
            continue
        if (
            isinstance(seq, list)
            and len(seq) == 21
            and all(
                isinstance(x, str)
                and re.fullmatch(r"(?:sad|dis|fear|neu|joy|ten|ins)\d+", x, flags=re.I)
                for x in seq
            )
        ):
            mapping[sub] = seq

    missing = [s for s in range(1, 54) if s not in mapping]
    if missing:
        raise ValueError(
            "Could not parse 21-item sequence for: "
            + ", ".join(f"sub-{s:02d}" for s in missing)
        )
    return mapping


def category_from_code(code: str) -> str:
    c = code.lower()
    for prefix, category in PREFIX_TO_CATEGORY.items():
        if c.startswith(prefix):
            return category
    raise ValueError(f"Unrecognized stimulus code: {code}")


def repeated_block_labels(seq):
    """
    Convert 21-item sequence into 42 raw-row labels:
    chunk1 twice, chunk2 twice, ..., chunk7 twice.
    """
    out = []
    block_pair = []
    within_pair_block = []
    within_block_trial = []

    for pair_idx in range(7):
        chunk = seq[pair_idx * 3:(pair_idx + 1) * 3]
        for member in (1, 2):
            out.extend(chunk)
            block_pair.extend([pair_idx + 1] * 3)
            within_pair_block.extend([member] * 3)
            within_block_trial.extend([1, 2, 3])

    return out, block_pair, within_pair_block, within_block_trial


def build_table(root: Path):
    # NOTE: Although the public README describes sub-01..sub-54 as complete-trial
    # participants, the participant-specific sequence list available in the README
    # provides parsable sequences only for sub-01..sub-53. We therefore exclude
    # sub-54 rather than infer or guess its trial labels.
    readme = root / "README.md"
    if not readme.exists():
        alt = root / "Readme.md"
        if alt.exists():
            readme = alt
        else:
            raise FileNotFoundError("README.md / Readme.md not found at dataset root.")

    sequences = parse_full_subject_sequences(readme)
    rows = []

    for sub in range(1, 54):
        sid = f"sub-{sub:02d}"
        tsv = root / sid / "beh" / f"{sid}_task-emotion_beh.tsv"
        if not tsv.exists():
            raise FileNotFoundError(tsv)

        df = pd.read_csv(tsv, sep="\t")
        if len(df) != 42:
            raise ValueError(f"{sid}: expected 42 behavioral rows, found {len(df)}")

        for i, name in enumerate(RATING_NAMES, start=1):
            col = f"score_{i}"
            if col not in df.columns:
                raise ValueError(f"{sid}: missing {col}")
            df[name] = pd.to_numeric(df[col], errors="coerce")

        labels, pair_idx, pair_member, within_block = repeated_block_labels(sequences[sub])

        df["stimulus_code"] = labels
        df["emotion_category"] = [category_from_code(x) for x in labels]
        df["block_pair"] = pair_idx
        df["block_within_pair"] = pair_member
        df["trial_within_block"] = within_block

        # Context is intentionally unknown. One of the two blocks in each pair is
        # imagery and the other is video, but the order was randomized.
        df["context"] = pd.NA
        df["participant_id"] = sid

        rows.append(
            df[
                [
                    "participant_id", "trial_number",
                    "block_pair", "block_within_pair", "trial_within_block",
                    "stimulus_code", "emotion_category", "context",
                ] + RATING_NAMES
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    expected_rows = 53 * 42
    if len(out) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for 53 complete participants, found {len(out)}."
        )
    return out


def diagnostics(df: pd.DataFrame):
    print(f"Rows: {len(df):,}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Stimulus codes: {df['stimulus_code'].nunique()}")
    print(f"Categories: {df['emotion_category'].nunique()}")

    discrete = ["joy", "inspiration", "tenderness", "sadness", "fear", "disgust"]
    diffs = []

    for _, r in df[df["emotion_category"] != "neutral"].iterrows():
        target_col = TARGET_COLUMN[r["emotion_category"]]
        target = r[target_col]
        others = [r[x] for x in discrete if x != target_col]
        diffs.append((target, np.nanmean(others)))

    arr = np.asarray(diffs, dtype=float)
    print(
        "\nTarget-emotion audit (non-neutral): "
        f"target={np.nanmean(arr[:,0]):.3f}; "
        f"non-target={np.nanmean(arr[:,1]):.3f}; "
        f"difference={np.nanmean(arr[:,0] - arr[:,1]):.3f}"
    )

    means = (
        df.groupby("emotion_category")[["valence", "arousal", "liking"]]
        .mean()
        .round(3)
    )
    print("\nCategory means:")
    print(means.to_string())

    pos = df[df["emotion_category"].isin(
        ["joy", "inspiration", "tenderness"]
    )]["valence"].mean()
    neu = df[df["emotion_category"] == "neutral"]["valence"].mean()
    neg = df[df["emotion_category"].isin(
        ["sadness", "fear", "disgust"]
    )]["valence"].mean()

    print(
        "\nBroad valence validation check: "
        f"positive={pos:.3f}; neutral={neu:.3f}; negative={neg:.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("behavior_labels.csv"))
    args = ap.parse_args()

    df = build_table(args.dataset_root)
    df.to_csv(args.out, index=False)
    diagnostics(df)

    print(f"\nWrote: {args.out.resolve()}")
    print(
        "\nNOTE: context is deliberately left missing, and sub-54 is deliberately excluded "
        "because no participant-specific sequence was available in the public README. "
        "Do not use this table for imagery-vs-video analyses."
    )


if __name__ == "__main__":
    main()
