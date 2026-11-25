"""E5 calibration: choose each model's selective-steering policy from development data, then freeze it.

The procedure is fixed here, before the development generations exist, so that the choice cannot drift with the
results.  Steering coverage is never a target: it is whatever the constraints below leave.

Data: one row per development message with the knowledge-graph-only reply (E3) and, per strength multiplier m
(1.0 = E4's state-scaled alpha, 0.5 = half of it), the steered candidate and the unsteered row of its batch, all
rated by both judges (mean of the two judges is used throughout).

Simulation of a policy (``simulate``): for each message, the four E5 steps are replayed offline - safety gate, context
cell, deficit gate on the batch's unsteered row, acceptance checks on each candidate in the cell's order - and the
change of every rating against E3 is recorded (0 when E5 does not generate candidates; the judged batch-unsteered
reply when it generates them but sends none, 0 if that row was not judged).

Selection (``fit``), per model:
1. Cells are (strategy, language) where the development set has at least ``MIN_N`` eligible messages of that
   strategy in that language; the remaining messages of a language form one residual cell ``*|<language>`` (rare
   strategies, and languages with little development data such as Hindi).  For each cell and each option - one
   strength multiplier, 1.0, 0.75 or 0.5 of E4's state-scaled alpha - the per-message changes are averaged and shrunk
   toward the model's language-level mean of the same option with a prior weight of ``PRIOR_WEIGHT`` messages
   (empirical-Bayes style), which guards against picking a cell for a lucky small sample.
2. An option is admissible for a cell when (shrunk) tone gain >= ``MIN_TONE_GAIN``, every quality metric and the
   validation rating >= -``CELL_QUALITY_FLOOR``, safety >= -``CELL_SAFETY_FLOOR``, the raw share of harmed sent
   replies <= ``MAX_HARM_SHARE`` and the cell has >= ``MIN_N`` messages.  The admissible option with the largest
   shrunk tone gain is chosen; none admissible -> the cell is not steered.
3. Population check: over every development message of the model (steered or not), the 95% bootstrap lower bound
   of each quality / validation change must be > -MARGIN_QUALITY and of safety > -MARGIN_SAFETY.  If not, the
   steered cell with the worst safety (then quality) contribution is dropped, until the check passes.
4. Global settings (deficit threshold, word-ratio cap) are chosen from a small grid by 5-fold cross-validation over
   messages: steps 1-3 are fitted on four folds and applied to the fifth; the setting with the largest
   out-of-fold tone gain whose out-of-fold population check passes is kept (ties: the simpler setting).
5. The chosen setting is refitted on all development messages and frozen (``results/e5_selective/policies``).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from chatbot_feature.emotion_reasoner.selective import PolicyCell, SelectivePolicy, accept

from .e5_common import MARGIN_QUALITY, MARGIN_SAFETY, QUALITY, SAFETY, SEED

MIN_TONE_GAIN = 0.15          # smallest tone gain (1-5 judge scale) worth an intervention
CELL_QUALITY_FLOOR = 0.05     # per-cell floor on each quality metric (half the population margin)
CELL_SAFETY_FLOOR = 0.05      # per-cell floor on safety (the population safety margin)
MAX_HARM_SHARE = 0.20         # share of sent steered replies with a quality / safety drop > 0.5 (as in E5-old)
MIN_N = 8                     # messages a cell needs before it can be steered (as in E5-old)
PRIOR_WEIGHT = 10.0           # shrinkage prior, in messages
OPTIONS = ((1.0,), (0.75,), (0.5,))   # one strength per cell (a fallback ladder such as (1.0, 0.5) is
# supported but not searched: the audit found the acceptance checks reject < 5% of candidates, so a ladder adds little)
GRID = {"deficit_threshold": (None, 0.30, 0.10), "max_word_ratio": (None, 2.0)}
FOLDS = 5
RESAMPLES = 2000
CONSTRAINED = (*QUALITY, "validation")      # metrics held to the quality margin (validation when rated)
HARM_CHECKS = (*QUALITY, SAFETY)


# ------------------------------------------------------------------------------------------------ simulate

def _metrics(table: pd.DataFrame) -> list[str]:
    return [m for m in ("judge_emotion", *QUALITY, SAFETY, "validation", "strategy_adherence") if f"kg_{m}" in table]


def acceptance_flags(table: pd.DataFrame, multiplier: float, settings: dict[str, Any], model: str) -> pd.Series:
    """Whether E5's acceptance checks pass each message's candidate at ``multiplier`` (False when absent)."""
    policy = SelectivePolicy(model=model)
    policy.acceptance["max_word_ratio"] = settings.get("max_word_ratio")
    key = _key(multiplier)
    out = []
    for unsteered, candidate, user in zip(table[f"b{key}_text"], table[f"c{key}_text"], table.text):
        if not isinstance(candidate, str) or not candidate:
            out.append(False)
            continue
        base = unsteered if isinstance(unsteered, str) and unsteered else candidate
        out.append(accept(policy, base, candidate, user_text=user)[0])
    return pd.Series(out, index=table.index)


def _key(multiplier: float) -> str:
    return f"{int(round(multiplier * 100)):03d}"


def simulate(table: pd.DataFrame, option: Sequence[float], settings: dict[str, Any], model: str,
             flags: dict[float, pd.Series] | None = None) -> pd.DataFrame:
    """Per-message rating changes against E3 if E5 used ``option`` for every message of ``table``.

    Columns ``d_<metric>``, ``steered`` (a candidate was sent), ``attempted`` (candidates generated), ``alpha_mult``.
    Messages that fail the safety / panel gate are never attempted.
    """
    metrics = _metrics(table)
    out = pd.DataFrame(index=table.index)
    for metric in metrics:
        out[f"d_{metric}"] = 0.0
    out["steered"] = False
    out["attempted"] = table.eligible.astype(bool) & bool(option)
    out["alpha_mult"] = 0.0
    out["harmed"] = False
    if not option:
        return out
    threshold = settings.get("deficit_threshold")
    blocked = pd.Series(False, index=table.index)
    if threshold is not None:
        blocked = (table.lang3 == "english") & (table.s0_batch >= threshold)
    remaining = out.attempted & ~blocked
    flags = flags or {}
    for multiplier in option:
        key = _key(multiplier)
        if f"c{key}_text" not in table:
            continue
        ok = flags.get(multiplier)
        if ok is None:
            ok = acceptance_flags(table, multiplier, settings, model)
        rated = table[f"c{key}_judge_emotion"].notna()
        take = remaining & ok & rated
        for metric in metrics:
            out.loc[take, f"d_{metric}"] = table.loc[take, f"c{key}_{metric}"] - table.loc[take, f"kg_{metric}"]
        out.loc[take, "steered"] = True
        out.loc[take, "alpha_mult"] = multiplier
        remaining &= ~take
    # Attempted but nothing sent: the batch's unsteered row is sent (its rating if judged, else no change).
    fallback = out.attempted & ~out.steered
    for multiplier in option:
        key = _key(multiplier)
        if f"b{key}_judge_emotion" in table:
            rated = fallback & table[f"b{key}_judge_emotion"].notna()
            for metric in metrics:
                if f"b{key}_{metric}" in table:
                    out.loc[rated, f"d_{metric}"] = table.loc[rated, f"b{key}_{metric}"] - table.loc[rated, f"kg_{metric}"]
            fallback &= ~rated
    checks = [f"d_{m}" for m in HARM_CHECKS if f"d_{m}" in out]
    out["harmed"] = out.steered & (out[checks].min(axis=1) < -0.5)
    return out


# ---------------------------------------------------------------------------------------------------- fit

def _bootstrap_low(values: np.ndarray, rng: np.random.Generator) -> float:
    values = values[~np.isnan(values)]
    if not len(values):
        return float("nan")
    means = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(1)
    return float(np.quantile(means, 0.025))


def population_check(effects: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Lower 95% bootstrap bounds of the mean change over all messages, and whether every constraint holds."""
    rng = np.random.default_rng(seed)
    bounds, passed = {}, True
    for metric in (*CONSTRAINED, SAFETY):
        column = f"d_{metric}"
        if column not in effects:
            continue
        low = _bootstrap_low(effects[column].to_numpy(dtype=float), rng)
        margin = MARGIN_SAFETY if metric == SAFETY else MARGIN_QUALITY
        bounds[metric] = {"mean": float(effects[column].mean()), "ci_low": low, "margin": -margin,
                          "passes": bool(low > -margin)}
        passed &= bounds[metric]["passes"]
    return {"passes": bool(passed), "bounds": bounds, "tone_gain": float(effects["d_judge_emotion"].mean()),
            "steered_share": float(effects.steered.mean()), "attempted_share": float(effects.attempted.mean())}


def _cell_estimates(table: pd.DataFrame, effects: pd.DataFrame, cells: pd.Series, languages: pd.Series) -> pd.DataFrame:
    """Raw and shrunk per-cell means of every change (shrunk toward the language mean of the same option)."""
    columns = [column for column in effects if column.startswith("d_")]
    frame = effects[columns].copy()
    frame["cell"], frame["language"] = cells, languages
    eligible = table.eligible.astype(bool)
    frame = frame[eligible]
    language_mean = frame.groupby("language")[columns].mean()
    grouped = frame.groupby("cell")
    raw = grouped[columns].mean()
    n = grouped.size()
    language_of_cell = grouped.language.first()
    prior = language_mean.loc[language_of_cell.to_numpy()].set_index(raw.index)
    weight = (n / (n + PRIOR_WEIGHT)).to_numpy()[:, None]
    shrunk = raw * weight + prior * (1 - weight)
    out = shrunk.add_prefix("shrunk_").join(raw.add_prefix("raw_"))
    out["n"] = n
    steered = effects.loc[eligible & effects.steered]
    out["sent"] = steered.groupby(frame.loc[steered.index, "cell"]).size().reindex(out.index).fillna(0).astype(int)
    out["harmed_share"] = (steered.harmed.groupby(frame.loc[steered.index, "cell"]).mean()
                           .reindex(out.index).fillna(0.0))
    return out


def admissible(row: pd.Series) -> tuple[bool, list[str]]:
    reasons = []
    if row.n < MIN_N:
        reasons.append(f"only {int(row.n)} development messages (needs {MIN_N})")
    if row.shrunk_d_judge_emotion < MIN_TONE_GAIN:
        reasons.append(f"tone gain {row.shrunk_d_judge_emotion:+.2f} < {MIN_TONE_GAIN}")
    for metric in CONSTRAINED:
        column = f"shrunk_d_{metric}"
        if column in row and pd.notna(row[column]) and row[column] < -CELL_QUALITY_FLOOR:
            reasons.append(f"{metric} {row[column]:+.2f} < -{CELL_QUALITY_FLOOR}")
    if row.shrunk_d_safety < -CELL_SAFETY_FLOOR:
        reasons.append(f"safety {row.shrunk_d_safety:+.2f} < -{CELL_SAFETY_FLOOR}")
    if row.harmed_share > MAX_HARM_SHARE:
        reasons.append(f"harmed share {row.harmed_share:.2f} > {MAX_HARM_SHARE}")
    return not reasons, reasons


def cell_keys(table: pd.DataFrame, known: Iterable[str] | None = None) -> pd.Series:
    """Cell of every message: ``strategy:<s>|<language>`` when that cell is known (given) or, when fitting, has at least
    ``MIN_N`` eligible messages; otherwise the language's residual cell ``*|<language>``."""
    strategy = "strategy:" + table.strategy.astype(str) + "|" + table.lang3.astype(str)
    if known is None:
        counts = strategy[table.eligible.astype(bool)].value_counts()
        known = set(counts[counts >= MIN_N].index)
    known = set(known)
    return pd.Series([key if key in known else f"*|{language}" for key, language in zip(strategy, table.lang3)],
                     index=table.index)


def fit_cells(table: pd.DataFrame, settings: dict[str, Any], model: str,
              options: Iterable[tuple[float, ...]] = OPTIONS) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    """Steps 1-3 for one setting: the chosen option per cell and the evidence behind every choice."""
    cells = cell_keys(table)
    available = [option for option in options if all(f"c{_key(m)}_text" in table for m in option)]
    flags = {m: acceptance_flags(table, m, settings, model) for m in {m for option in available for m in option}}
    estimates = {option: _cell_estimates(table, simulate(table, option, settings, model, flags), cells, table.lang3)
                 for option in available}
    chosen: dict[str, tuple[float, ...]] = {}
    evidence: dict[str, Any] = {}
    for cell in sorted(set(cells[table.eligible.astype(bool)])):
        best, best_gain, notes = (), -np.inf, {}
        for option, frame in estimates.items():
            if cell not in frame.index:
                continue
            row = frame.loc[cell]
            ok, reasons = admissible(row)
            notes[",".join(map(str, option))] = {
                "admissible": ok, "reasons": reasons, "n": int(row.n), "sent": int(row.sent),
                "harmed_share": round(float(row.harmed_share), 3),
                **{key: round(float(row[key]), 4) for key in row.index if key.startswith(("shrunk_", "raw_")) and pd.notna(row[key])}}
            if ok and row.shrunk_d_judge_emotion > best_gain:
                best, best_gain = option, row.shrunk_d_judge_emotion
        chosen[cell] = best
        evidence[cell] = notes
    # Step 3: population check, pruning the steered cell with the worst safety, then quality, contribution.
    pruned = []
    while True:
        effects = apply_cells(table, chosen, settings, model, flags)
        check = population_check(effects)
        if check["passes"] or not any(chosen.values()):
            break
        contributions = []
        for cell, option in chosen.items():
            if not option:
                continue
            members = (cells == cell) & effects.steered
            safety = float(effects.loc[members, f"d_{SAFETY}"].sum())
            quality = float(effects.loc[members, [f"d_{m}" for m in CONSTRAINED if f"d_{m}" in effects]].sum().min())
            contributions.append((safety, quality, cell))
        _, _, worst = min(contributions)
        pruned.append(worst)
        chosen[worst] = ()
    return chosen, {"cells": evidence, "population": check, "pruned_for_population": pruned}


def apply_cells(table: pd.DataFrame, chosen: dict[str, tuple[float, ...]], settings: dict[str, Any], model: str,
                flags: dict[float, pd.Series] | None = None) -> pd.DataFrame:
    """Per-message changes when each cell uses its chosen option (unknown cells are not steered)."""
    if table.empty:
        return simulate(table, (), settings, model)
    cells = cell_keys(table, known=[key for key in chosen if key.startswith("strategy:")])
    parts = []
    for option in set(chosen.values()) | {()}:
        members = cells.map(lambda cell: chosen.get(cell, ()) == option)
        if members.any():
            parts.append(simulate(table[members], option, settings, model,
                                  {m: f[members] for m, f in (flags or {}).items()}))
    return pd.concat(parts).loc[table.index]


def cross_validate(table: pd.DataFrame, settings: dict[str, Any], model: str, folds: int = FOLDS) -> dict[str, Any]:
    """Out-of-fold effects of the whole selection procedure (steps 1-3) for one setting."""
    rng = np.random.default_rng(SEED)
    strata = table.lang3.astype(str) + "|" + table.strategy.astype(str)
    fold = pd.Series(0, index=table.index)
    for _, members in table.groupby(strata).groups.items():
        order = rng.permutation(len(members))
        fold.loc[members[order]] = np.arange(len(members)) % folds
    parts = []
    for k in range(folds):
        train, test = table[fold != k], table[fold == k]
        chosen, _ = fit_cells(train, settings, model)
        parts.append(apply_cells(test, chosen, settings, model))
    effects = pd.concat(parts).loc[table.index]
    return {**population_check(effects), "effects": effects}


def fit(table: pd.DataFrame, model: str, alpha_range: tuple[float, float], *,
        grid: dict[str, Sequence[Any]] | None = None, source: dict[str, Any] | None = None) -> tuple[SelectivePolicy, dict[str, Any]]:
    """Steps 1-5: the frozen policy for ``model`` and the calibration report."""
    grid = grid or GRID
    settings_list = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    results = []
    for settings in settings_list:
        cv = cross_validate(table, settings, model)
        complexity = sum(value is not None for value in settings.values())
        results.append({"settings": settings, "cv_tone_gain": cv["tone_gain"], "cv_passes": cv["passes"],
                        "cv_steered_share": cv["steered_share"], "cv_bounds": cv["bounds"], "complexity": complexity})
    passing = [row for row in results if row["cv_passes"]]
    if passing:
        best = max(passing, key=lambda row: (round(row["cv_tone_gain"], 3), -row["complexity"]))
        selection = "largest out-of-fold tone gain among settings whose out-of-fold population check passed"
    else:
        best = min(results, key=lambda row: row["complexity"])
        selection = "no setting passed the out-of-fold population check; the simplest setting is refitted (its cells " \
                    "are still subject to the in-sample check, which may leave nothing steered)"
    chosen, evidence = fit_cells(table, best["settings"], model)
    policy = SelectivePolicy(model=model, version="e5-1", frozen=True, alpha_range=tuple(alpha_range))
    for cell, option in chosen.items():
        notes = evidence["cells"][cell]
        reason = ("steered: " + ", ".join(f"x{m:g}" for m in option) + " of E4's state-scaled strength; shrunk tone "
                  f"gain {notes[','.join(map(str, option))]['shrunk_d_judge_emotion']:+.2f}") if option else \
            "not steered: " + "; ".join(sorted({r for opt in notes.values() for r in opt["reasons"]})[:4] or
                                        ["pruned by the population check"])
        policy.cells[cell] = PolicyCell(multipliers=tuple(option), reason=reason, evidence=notes)
    threshold = best["settings"].get("deficit_threshold")
    policy.deficit = {"enabled": threshold is not None, "signal": "gate_target", "threshold": threshold,
                      "languages": ["english"]}
    policy.acceptance["max_word_ratio"] = best["settings"].get("max_word_ratio")
    policy.provenance = {
        "fitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "procedure": "generation_analysis.e5_fit (see module docstring)",
        "constants": {"MIN_TONE_GAIN": MIN_TONE_GAIN, "CELL_QUALITY_FLOOR": CELL_QUALITY_FLOOR,
                      "CELL_SAFETY_FLOOR": CELL_SAFETY_FLOOR, "MAX_HARM_SHARE": MAX_HARM_SHARE, "MIN_N": MIN_N,
                      "PRIOR_WEIGHT": PRIOR_WEIGHT, "MARGIN_QUALITY": MARGIN_QUALITY, "MARGIN_SAFETY": MARGIN_SAFETY,
                      "OPTIONS": [list(o) for o in OPTIONS], "GRID": {k: list(v) for k, v in grid.items()},
                      "FOLDS": FOLDS},
        "development": source or {}, "messages": int(len(table)), "eligible": int(table.eligible.sum()),
        "selected_settings": best["settings"], "selection_rule": selection,
        "in_sample_population": evidence["population"], "pruned_for_population": evidence["pruned_for_population"],
        "cross_validation": [{key: value for key, value in row.items()} for row in results],
    }
    report = {"model": model, "settings": best["settings"], "selection": selection, "grid": results,
              "population_in_sample": evidence["population"], "pruned": evidence["pruned_for_population"],
              "steered_cells": {cell: list(option) for cell, option in chosen.items() if option},
              "cells": evidence["cells"]}
    return policy, report


def from_development(dev: pd.DataFrame) -> pd.DataFrame:
    """Calibration rows from the earlier studies (``e5_common.development_table``): the E4 candidate is the x1.0
    candidate.  It was rated only when E4 sent it; when E4 rejected it, E4 sent (and the judges rated) the batch's
    unsteered row instead, and E5 - whose checks include E4's - would have rejected it too."""
    from chatbot_feature.emotion_reasoner.risk_screen import ALLOWED
    metrics = ("judge_emotion", *QUALITY, SAFETY, "validation", "strategy_adherence")
    out = dev[["model", "study", "item_id", "group", "text", "strategy", "target", "lang3", "language", "turns",
               "user_emotion", "intensity", "valence", "risk_category", "safety_flag", "state_level", "s0_batch",
               "e4_applied", "e4_planned"]].copy()
    out["eligible"] = dev.risk_category.isin(ALLOWED) & ~dev.safety_flag.astype(bool) & (dev.e4_planned > 0)
    out["c100_text"] = dev.candidate
    out["b100_text"] = dev.batch0
    applied = dev.e4_applied > 0
    for metric in metrics:
        if f"kg_{metric}_both" not in dev:
            continue
        out[f"kg_{metric}"] = dev[f"kg_{metric}_both"]
        out[f"c100_{metric}"] = dev[f"e4_{metric}_both"].where(applied)
        batch_rated = dev[f"e4_{metric}_both"].where(~applied & dev.batch0.notna())
        out[f"b100_{metric}"] = batch_rated.fillna(dev[f"kg_{metric}_both"].where(dev.batch0_is_kg))
    out.index = out.study + ":" + out.item_id
    return out


def table_fingerprint(table: pd.DataFrame) -> str:
    text = table.drop(columns=[c for c in table.columns if c == "history"], errors="ignore").to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["MIN_TONE_GAIN", "OPTIONS", "GRID", "simulate", "fit", "fit_cells", "apply_cells", "cross_validate",
           "population_check", "acceptance_flags", "table_fingerprint", "from_development", "cell_keys"]
