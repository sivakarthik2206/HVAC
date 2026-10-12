"""SageMaker inference handler: one PLC record in, ranked valve set points out.

The endpoint does two things the training job cannot: it takes the live ambient
conditions, and it searches valve combinations for the lowest predicted power.

Contract
--------
POST  application/json

    {
      "outdoor_temp": 34.5,         # Outdoor-T8  °C      (required)
      "outdoor_humidity": 55.0,     # Outdoor RH  %       (required)
      "room_temp": 22.0,            # Room-T7     °C      (required)
      "indoor_humidity": 48.0,      # Indoor RH   %       (required)
      "search": "observed",         # "observed" (default) | "full"
      "top_k": 5
    }

Keys may also be the raw PLC column names ("Outdoor-T8(°C)" etc.), so Lambda 2
can forward a record unchanged.

Response

    {
      "recommended": {"PCV-1(%)": 100, ..., "predicted_kw": 3.677,
                      "measured_kw": 3.666, "rows": 680, "observed": true},
      "alternatives": [...],
      "runner_up_gap_kw": 0.0224,
      "model_error_kw": 0.0631,
      "separable": false,
      "actionable": false,
      "status": "ADVISORY",
      "warnings": [...],
      "model_version": "...", "generated_at": "..."
    }

`search`:
  "observed" ranks only the valve combinations the plant has actually run.
  "full" ranks every arithmetic combination. Tree models do not extrapolate -
  outside the states they were trained on they return the nearest leaf - so
  every "full" candidate that was never run is flagged and the response carries
  a warning. "observed" is the default for that reason.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# friendly key -> PLC column name
ALIASES = {
    'outdoor_temp': 'Outdoor-T8(°C)',
    'outdoor_humidity': 'Outdoor Humidity(%)',
    'room_temp': 'Room-T7(°C)',
    'indoor_humidity': 'Indoor Humidity(%)',
}
MAX_TOP_K = 25


class BadRequest(ValueError):
    """400, not 500 - the caller can fix this."""


# --------------------------------------------------------------------------
# SageMaker hooks
# --------------------------------------------------------------------------
def model_fn(model_dir: str) -> dict:
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, 'model.json'))
    with open(os.path.join(model_dir, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)

    # pre-build the observed candidate table once per container
    observed = pd.DataFrame([
        {**c['valves'], 'measured_kw': c['measured_kw'],
         'rows': c['rows'], 'days': c['days']}
        for c in meta['observed_combinations']])

    log.info('loaded model: %d features, %d observed combinations, actionable=%s',
             len(meta['features']), len(observed), meta.get('actionable'))
    return {'booster': booster, 'meta': meta, 'observed': observed}


def input_fn(body, content_type='application/json') -> dict:
    if content_type and 'json' not in content_type:
        raise BadRequest(f'Unsupported content type {content_type}; send application/json')
    if isinstance(body, (bytes, bytearray)):
        body = body.decode('utf-8')
    try:
        payload = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError as exc:
        raise BadRequest(f'Body is not valid JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise BadRequest('Body must be a JSON object')
    return payload


def predict_fn(payload: dict, bundle: dict) -> dict:
    meta, booster = bundle['meta'], bundle['booster']
    valves, features = meta['valves'], meta['features']

    conditions = _read_conditions(payload, meta)
    search = str(payload.get('search', 'observed')).lower()
    if search not in ('observed', 'full'):
        raise BadRequest("'search' must be 'observed' or 'full'")
    top_k = int(payload.get('top_k', 5))
    if not 1 <= top_k <= MAX_TOP_K:
        raise BadRequest(f"'top_k' must be between 1 and {MAX_TOP_K}")

    grid, warnings = _candidates(bundle, search, valves)
    for col, val in conditions.items():
        grid[col] = val

    dm = xgb.DMatrix(grid[features].astype(float).values, feature_names=features)
    grid['predicted_kw'] = booster.predict(dm).round(4)
    grid = grid.sort_values('predicted_kw').reset_index(drop=True)

    warnings += _condition_warnings(conditions, meta)

    val = meta.get('validation', {})
    model_error = float(val.get('pooled_mae') or 0.0)
    gap = (float(grid['predicted_kw'].iloc[1] - grid['predicted_kw'].iloc[0])
           if len(grid) > 1 else None)
    separable = bool(gap is not None and model_error > 0 and gap > model_error)

    actionable = bool(meta.get('actionable')) and separable
    if not meta.get('actionable'):
        warnings.append(
            f"Model failed leave-one-day-out validation "
            f"(pooled R2 {val.get('pooled_r2', float('nan')):.3f}, "
            f"passed {val.get('days_passed')}/{val.get('n_days')} days). "
            f"Set points are advisory only.")
    if gap is not None and not separable:
        warnings.append(
            f"Top two candidates differ by {gap:.4f} kW, within the model's "
            f"{model_error:.4f} kW out-of-day error, so the ranking cannot "
            f"separate them.")

    keep = valves + ['predicted_kw', 'measured_kw', 'rows', 'days']
    rows = grid.head(top_k)[keep].to_dict('records')
    for r in rows:
        for v in valves:
            r[v] = int(r[v])
        r['predicted_kw'] = round(float(r['predicted_kw']), 4)
        if pd.isna(r.get('measured_kw')):
            r.update(measured_kw=None, rows=0, days=0, observed=False)
        else:
            r.update(measured_kw=round(float(r['measured_kw']), 4),
                     rows=int(r['rows']), days=int(r['days']), observed=True)

    return {
        'status': 'ACTIONABLE' if actionable else 'ADVISORY',
        'actionable': actionable,
        'recommended': rows[0],
        'alternatives': rows[1:],
        'runner_up_gap_kw': round(gap, 4) if gap is not None else None,
        'model_error_kw': round(model_error, 4),
        'separable': separable,
        'conditions': conditions,
        'search': search,
        'candidates_scored': int(len(grid)),
        'warnings': warnings,
        'model_version': meta.get('data', {}).get('days', ['unknown'])[-1],
        'training_rows': meta.get('training_rows'),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def output_fn(prediction: dict, accept='application/json'):
    return json.dumps(prediction), 'application/json'


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_conditions(payload: dict, meta: dict) -> dict:
    """Accept friendly keys or raw PLC column names."""
    out = {}
    for col in meta['conditions']:
        alias = next((a for a, c in ALIASES.items() if c == col), None)
        raw = payload.get(col, payload.get(alias))
        if raw is None:
            raise BadRequest(
                f"Missing '{alias or col}'. Required: "
                f"{sorted(a for a, c in ALIASES.items() if c in meta['conditions'])}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"'{alias or col}' must be a number") from exc
        if not np.isfinite(value):
            raise BadRequest(f"'{alias or col}' must be finite")
        out[col] = value
    return out


def _candidates(bundle: dict, search: str, valves: list) -> tuple[pd.DataFrame, list]:
    meta = bundle['meta']
    if search == 'observed':
        return bundle['observed'].copy(), []

    options = [meta['valve_options'][v] for v in valves]
    total_combinations = np.prod([len(opt) for opt in options])
    if total_combinations > 1000:
        raise BadRequest(f"Combinatorial search space too large ({total_combinations} candidates). Use search='observed'.")
    grid = pd.DataFrame(list(itertools.product(*options)), columns=valves)
    grid = grid.merge(bundle['observed'], on=valves, how='left')
    unseen = int(grid['measured_kw'].isna().sum())
    warn = []
    if unseen:
        warn.append(
            f"{unseen} of {len(grid)} candidates were never run on this plant. "
            f"XGBoost does not extrapolate, so their predictions carry no "
            f"information. Use search='observed' for a defensible ranking.")
    return grid, warn


def _condition_warnings(conditions: dict, meta: dict) -> list:
    warn = []
    for col, value in conditions.items():
        lo, hi = meta['condition_ranges'][col]
        if not lo <= value <= hi:
            warn.append(f'{col} = {value} is outside the trained range '
                        f'{lo:.1f} to {hi:.1f}; the model is extrapolating.')
    return warn
