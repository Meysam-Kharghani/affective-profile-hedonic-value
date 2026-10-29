#!/usr/bin/env python3
"""
Recover imagery/video context labels for EmoEEG-MC from EDF annotations in the dataset directory.

Context recovery uses participant-specific TypeID mappings and the published
paired three-trial block design. Behavioral ratings and hedonic outcomes are
not used to infer context.

Example:
    python src/data_integrity/02_recover_context_from_edf.py \
      --dataset-root "${DATASET_ROOT}" \
      --behavior "${OUTPUT_DIR}/behavior_labels.csv" \
      --outdir "${OUTPUT_DIR}/context"
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRIGGER_MAP = {
    1:(31,30), 2:(33,32), 3:(33,32), 4:(51,50), 5:(6,5),
    6:(24,23), 7:(51,50), 8:(42,41), 9:(6,5), 10:(24,23),
    11:(24,23), 12:(33,32), 13:(24,23), 14:(24,23), 15:(6,5),
    16:(24,23), 17:(6,5), 18:(6,5), 19:(6,5), 20:(6,5),
    21:(36,35), 22:(56,55), 23:(36,35), 24:(36,35), 25:(6,5),
    26:(6,5), 27:(6,5), 28:(16,15), 29:(26,25), 30:(26,25),
    31:(6,5), 32:(6,5), 33:(15,14), 34:(6,5), 35:(6,5),
    36:(6,5), 37:(6,5), 38:(16,15), 39:(9,8), 40:(6,5),
    41:(56,55), 42:(26,25), 43:(38,37), 44:(16,15), 45:(16,15),
    46:(10,9), 47:(10,9), 48:(6,5), 49:(6,5), 50:(16,15),
    51:(26,25), 52:(6,5), 53:(38,36),
}

AFFECTS = [
    "joy", "inspiration", "tenderness", "sadness",
    "fear", "disgust", "arousal", "familiarity",
]
OUTCOMES = ["valence", "liking"]


def legal_sequences():
    out = []
    for bits in itertools.product([0, 1], repeat=7):
        seq = []
        orders = []
        for bit in bits:
            if bit == 0:
                pair = ["video"] * 3 + ["imagery"] * 3
                orders.append("vid-ima")
            else:
                pair = ["imagery"] * 3 + ["video"] * 3
                orders.append("ima-vid")
            seq.extend(pair)
        out.append((tuple(seq), tuple(orders)))
    return out


LEGAL = legal_sequences()


def is_subsequence(short, long):
    """Whether sequence short occurs as an order-preserving subsequence of long."""
    it = iter(long)
    return all(any(x == y for y in it) for x in short)


def structural_candidates(observed):
    """
    If observed has extra markers: legal 42-seq must be a subsequence of observed.
    If observed has missing markers: observed must be a subsequence of legal 42-seq.
    If equal length: exact equality required.
    Return distinct legal target sequences.
    """
    obs = tuple(observed)
    matches = []
    for seq, orders in LEGAL:
        if len(obs) > 42:
            ok = is_subsequence(seq, obs)
        elif len(obs) < 42:
            ok = is_subsequence(obs, seq)
        else:
            ok = obs == seq
        if ok:
            matches.append((seq, orders))
    return matches


def collapse_consecutive(seq):
    out = []
    for x in seq:
        if not out or x != out[-1]:
            out.append(x)
    return out


def natural_key(p):
    m = re.search(r"run-(\d+)", p.name.lower())
    return (int(m.group(1)) if m else 0, p.name.lower())


def find_edfs(root, sid):
    eeg = root / sid / "eeg"
    if not eeg.exists():
        return [], []

    candidates = sorted(
        [p for p in eeg.glob("*.edf")
         if "task-emotion" in p.name.lower() and "_eeg.edf" in p.name.lower()],
        key=natural_key,
    )
    if not candidates:
        candidates = sorted(eeg.glob("*.edf"), key=natural_key)

    run_specific = [p for p in candidates if re.search(r"run-\d+", p.name.lower())]
    if run_specific:
        candidates = run_specific

    usable, bad = [], []
    for p in candidates:
        try:
            if p.stat().st_size <= 0:
                bad.append(f"{p.name}: zero-byte")
            else:
                usable.append(p)
        except OSError as e:
            bad.append(f"{p.name}: {e}")
    return usable, bad


def read_annotations(path):
    errs = []
    try:
        import pyedflib
        f = pyedflib.EdfReader(str(path))
        try:
            ons, dur, desc = f.readAnnotations()
            return [(float(o), float(d), str(s)) for o,d,s in zip(ons,dur,desc)], "pyedflib"
        finally:
            f.close()
    except Exception as e:
        errs.append(f"pyedflib={type(e).__name__}:{e}")

    try:
        import mne
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        return [
            (float(o), float(d), str(s))
            for o,d,s in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description
            )
        ], "mne"
    except Exception as e:
        errs.append(f"mne={type(e).__name__}:{e}")

    raise RuntimeError(f"{path}: " + " | ".join(errs))


def trigger_number(desc):
    # First prefer a description that is essentially just an integer.
    s = str(desc).strip()
    m = re.fullmatch(r"(?:typeid\s*[:=]?\s*|t\s*)?(-?\d+)(?:\.0+)?", s, flags=re.I)
    if m:
        return int(m.group(1))

    # Then common event-label forms.
    for pat in [
        r"(?:stimulus[/ _-]*s|typeid|trigger|event|code)\s*[:=/ _-]*(-?\d+)\b",
        r"\bs\s*([0-9]+)\b",
    ]:
        m = re.search(pat, s, flags=re.I)
        if m:
            return int(m.group(1))

    # Compatibility fallback for alternative annotation labels.
    nums = re.findall(r"(?<!\d)-?\d+(?!\d)", s)
    return int(nums[-1]) if nums else None


def extract_contexts(annotations, vid, ima, mode):
    seq = []
    events = []
    for file_idx, onset, dur, desc in annotations:
        c = trigger_number(desc)
        if c is None:
            continue
        if mode == "exact":
            if c == vid:
                ctx = "video"
            elif c == ima:
                ctx = "imagery"
            else:
                continue
        else:
            if c % 10 == vid % 10:
                ctx = "video"
            elif c % 10 == ima % 10:
                ctx = "imagery"
            else:
                continue
        seq.append(ctx)
        events.append((file_idx, onset, dur, desc, c, ctx))
    return seq, events


def resolve_sequence(observed):
    """
    Return (target_seq, pair_orders, method, n_candidates) or None.
    Preference:
      direct legal sequence
      unique structural repair of raw sequence
      unique structural repair after consecutive-collapse
    """
    obs = list(observed)

    raw_matches = structural_candidates(obs)
    if len(obs) == 42 and len(raw_matches) == 1:
        seq, orders = raw_matches[0]
        return list(seq), list(orders), "direct", 1

    if len(raw_matches) == 1:
        seq, orders = raw_matches[0]
        method = "structural_prune" if len(obs) > 42 else "structural_complete"
        return list(seq), list(orders), method, 1

    collapsed = collapse_consecutive(obs)
    if collapsed != obs:
        col_matches = structural_candidates(collapsed)
        if len(col_matches) == 1:
            seq, orders = col_matches[0]
            if len(collapsed) == 42:
                method = "collapse_direct"
            elif len(collapsed) > 42:
                method = "collapse_structural_prune"
            else:
                method = "collapse_structural_complete"
            return list(seq), list(orders), method, 1
        return None, len(raw_matches), len(col_matches), len(collapsed)

    return None, len(raw_matches), None, len(collapsed)


def subject_recovery(root, n):
    sid = f"sub-{n:02d}"
    vid, ima = TRIGGER_MAP[n]
    files, bad_files = find_edfs(root, sid)
    if not files:
        return {
            "ok": False, "participant_id": sid, "reason": "no_usable_edf",
            "bad_files": "|".join(bad_files), "vid": vid, "ima": ima,
        }

    annotations = []
    backends = []
    details = []
    for fi, p in enumerate(files):
        try:
            a, backend = read_annotations(p)
        except Exception as e:
            bad_files.append(str(e))
            continue
        backends.append(backend)
        details.append(f"{p.name}:{len(a)}ann")
        for onset, dur, desc in sorted(a, key=lambda z: z[0]):
            annotations.append((fi, onset, dur, desc))

    if not annotations:
        return {
            "ok": False, "participant_id": sid, "reason": "no_readable_annotations",
            "bad_files": "|".join(bad_files), "vid": vid, "ima": ima,
        }

    diagnostics = []
    for mode in ("exact", "mod10"):
        obs, events = extract_contexts(annotations, vid, ima, mode)
        res = resolve_sequence(obs)

        if isinstance(res, tuple) and len(res) == 4 and isinstance(res[0], list):
            seq, orders, method, ncan = res
            return {
                "ok": True,
                "participant_id": sid,
                "sequence": seq,
                "pair_orders": orders,
                "trigger_mode": mode,
                "repair_method": method,
                "n_raw_context_events": len(obs),
                "n_video_raw": obs.count("video"),
                "n_imagery_raw": obs.count("imagery"),
                "n_context_events_used": 42,
                "vid": vid,
                "ima": ima,
                "n_edf_files": len(files),
                "edf_files": "|".join(p.name for p in files),
                "bad_files": "|".join(bad_files),
                "backend": "|".join(sorted(set(backends))),
                "details": "|".join(details),
            }

        if isinstance(res, tuple):
            _, raw_n, col_n, col_len = res
        else:
            raw_n = col_n = col_len = None
        diagnostics.append(
            f"{mode}:raw={len(obs)} "
            f"(v={obs.count('video')},i={obs.count('imagery')}),"
            f"raw_legal_candidates={raw_n},collapsed_len={col_len},"
            f"collapsed_candidates={col_n}"
        )

    # Numeric code frequency helps diagnose remaining hard subjects.
    nums = [trigger_number(d) for _,_,_,d in annotations]
    nums = [x for x in nums if x is not None]
    freq = pd.Series(nums, dtype="int64").value_counts().head(20).to_dict()

    return {
        "ok": False,
        "participant_id": sid,
        "reason": "ambiguous_or_incomplete",
        "diagnostic": " || ".join(diagnostics),
        "top_codes": repr(freq),
        "bad_files": "|".join(bad_files),
        "vid": vid, "ima": ima,
        "n_edf_files": len(files),
        "edf_files": "|".join(p.name for p in files),
        "backend": "|".join(sorted(set(backends))),
        "details": "|".join(details),
    }


def complete_case(df):
    cols = AFFECTS + OUTCOMES
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.loc[~num.isna().any(axis=1)].copy().reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--behavior", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    behavior = pd.read_csv(args.behavior)
    if behavior["participant_id"].nunique() != 53:
        raise ValueError("Expected 53 participants in the behavior table.")
    if len(behavior) != 2226:
        raise ValueError(f"Expected 2226 rows, got {len(behavior)}.")

    recovered = {}
    audit_rows = []

    print("=== CONTEXT RECOVERY ===")
    for n in range(1,54):
        r = subject_recovery(args.dataset_root, n)
        sid = r["participant_id"]
        if r["ok"]:
            recovered[sid] = r
            print(
                f"{sid}: PASS  trigger={r['trigger_mode']:<5} "
                f"repair={r['repair_method']:<28} "
                f"raw={r['n_raw_context_events']:>3} "
                f"pairs={','.join(r['pair_orders'])}"
            )
        else:
            print(
                f"{sid}: UNRESOLVED  reason={r.get('reason')}  "
                f"{r.get('diagnostic','')}  "
                f"bad_files={r.get('bad_files','')}"
            )

        row = {k:v for k,v in r.items() if k not in {"sequence", "pair_orders", "ok"}}
        row["audit_pass"] = bool(r["ok"])
        if r["ok"]:
            row["pair_orders"] = "|".join(r["pair_orders"])
            row["n_video_final"] = r["sequence"].count("video")
            row["n_imagery_final"] = r["sequence"].count("imagery")
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.outdir / "context_recovery_audit.csv"
    audit.to_csv(audit_path, index=False)

    # Build partial dataset from all independently/uniquely recovered participants.
    parts = []
    for sid, g in behavior.groupby("participant_id", sort=False):
        if sid not in recovered:
            continue
        if len(g) != 42:
            raise ValueError(f"{sid}: expected 42 behavior rows, got {len(g)}")
        gg = g.copy()
        rr = recovered[sid]
        gg["context"] = rr["sequence"]
        gg["context_source"] = (
            f"edf_{rr['trigger_mode']}_{rr['repair_method']}"
        )
        parts.append(gg)

    merged = pd.concat(parts, ignore_index=True)
    merged_path = args.outdir / "behavior_with_context.csv"
    merged.to_csv(merged_path, index=False)

    cc = complete_case(merged)
    canonical = pd.DataFrame({
        "canonical_row_index": np.arange(len(cc)),
        "subject_id": cc["participant_id"].astype(str),
        "subject_number": cc["participant_id"].str.extract(r"(\d+)")[0].astype(int),
        "stimulus_id": cc["stimulus_code"].astype(str),
        "context": cc["context"].astype(str),
        "emotion_category": cc["emotion_category"].astype(str),
        "context_source": cc["context_source"].astype(str),
        "is_analysis_eligible": True,
    })
    for c in AFFECTS + OUTCOMES:
        canonical[c] = pd.to_numeric(cc[c], errors="raise")

    canonical_path = args.outdir / "canonical_context.csv"
    canonical.to_csv(canonical_path, index=False)

    print("\n=== SUMMARY ===")
    print(f"Recovered participants: {len(recovered)}/53")
    print(f"Unresolved participants: {53-len(recovered)}")
    print(f"Recovered rows before complete-case: {len(merged)}")
    print(f"Complete-case rows: {len(canonical)}")
    print("\nRecovery methods:")
    if len(recovered):
        print(
            audit.loc[audit.audit_pass, "repair_method"]
            .value_counts().to_string()
        )
    print("\nContext counts in complete-case subset:")
    print(canonical["context"].value_counts().to_string())
    print("\nUnresolved IDs:")
    print(", ".join(audit.loc[~audit.audit_pass, "participant_id"].astype(str)))
    print(f"\nWrote: {audit_path.resolve()}")
    print(f"Wrote: {merged_path.resolve()}")
    print(f"Wrote: {canonical_path.resolve()}")
    print(
        "\nNOTE: structural repair uses only the prespecified experimental block "
        "design and EDF trigger order. Ambiguous subjects remain excluded."
    )


if __name__ == "__main__":
    main()
