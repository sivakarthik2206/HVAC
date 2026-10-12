"""SageMaker training entry point: HVAC energy model + valve optimiser.

Runs inside the SageMaker XGBoost framework container. Reads every PLC export
S3 put under the `training` channel, repairs the known data faults, trains an
XGBoost regressor on valve positions + ambient conditions, and validates it
LEAVE-ONE-DAY-OUT.

The validation result is written into the model artifact. The inference handler
reads it and refuses to present set points as actionable when it fails, so a
bad model cannot silently reach the dashboard.

Artifact written to /opt/ml/model:
    model.json          XGBoost booster
    metadata.json       features, valve options, observed combinations,
                        condition ranges, validation metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

VALVES = ['PCV-1(%)', 'PCV-2(%)', 'PCV-3(%)', 'PCV-4(%)']
CONDITIONS = ['Outdoor-T8(°C)', 'Outdoor Humidity(%)',
              'Room-T7(°C)', 'Indoor Humidity(%)']
POWER_PARTS = ['Comp Power(KW)', 'Indoor Power(KW)', 'Fan Power(KW)']
TARGET = 'Total kW'
SETTLE_SECONDS = 600
COMP_ON_KW = 0.3
SEED = 42

# a model must clear both of these to be marked actionable
MIN_POOLED_R2 = 0.50
MIN_DAY_PASS_RATE = 0.70


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def read_any(path: Path) -> pd.DataFrame | None:
    """Accept whatever Lambda 1 wrote: xlsx, csv, parquet or json lines."""
    try:
        if path.suffix.lower() in ('.xlsx', '.xlsm'):
            return pd.read_excel(path)
        if path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        if path.suffix.lower() in ('.parquet', '.pq'):
            return pd.read_parquet(path)
        if path.suffix.lower() in ('.json', '.jsonl', '.ndjson'):
            return pd.read_json(path, lines=path.suffix.lower() != '.json')
    except Exception as exc:                                   # noqa: BLE001
        log.warning('skipping %s: %s', path.name, exc)
    return None


def load(folder: str) -> tuple[pd.DataFrame, dict]:
    paths = [p for p in sorted(Path(folder).rglob('*')) if p.is_file()]
    frames, relabelled, skipped = [], [], []
    for p in paths:
        d = read_any(p)
        if d is None or not {'Time', 'Comp Power(KW)'} <= set(d.columns):
            skipped.append(p.name)
            continue
        d['src'] = p.name

        # FAULT 1 - older exports name the hot discharge sensor 'Comp In-T1'.
        # Detect by temperature; never trust the header.
        if {'Comp In-T1(°C)', 'Comp Out-T4(°C)'} <= set(d.columns):
            s_in = d['Comp In-T1(°C)'].dropna()
            s_out = d['Comp Out-T4(°C)'].dropna()
            if len(s_in) > 0 and len(s_out) > 0 and s_in.mean() > s_out.mean() + 5:
                d = d.rename(columns={'Comp In-T1(°C)': '_t',
                                      'Comp Out-T4(°C)': 'Comp In-T1(°C)'})
                d = d.rename(columns={'_t': 'Comp Out-T4(°C)'})
                relabelled.append(p.name)
        frames.append(d)

    if not frames:
        sys.exit(f'No readable PLC files under {folder}. Skipped: {skipped[:10]}')

    df = pd.concat(frames, ignore_index=True)
    df['Time'] = pd.to_datetime(df['Time'])

    # FAULT 2 - the shipped Total Power column omits the compressor.
    # Require all sub-meters to be valid so NaNs are preserved rather than summed as 0.
    df = df.drop(columns=['Total Power(KW)'], errors='ignore')
    df[TARGET] = df[POWER_PARTS].sum(axis=1, min_count=len(POWER_PARTS))

    # FAULT 3 - overlapping re-downloads.
    n0 = len(df)
    df = df.sort_values('Time').drop_duplicates('Time').reset_index(drop=True)

    df['date'] = df['Time'].dt.date
    df['comp_on'] = df['Comp Power(KW)'] > COMP_ON_KW

    # Settling calculation: reset blocks on valve state changes OR time gaps > 60s
    key = df[VALVES].astype(str).agg('|'.join, axis=1)
    time_gap = df['Time'].diff() > pd.Timedelta(seconds=60)
    block = ((key != key.shift()) | time_gap).cumsum()
    held = df.groupby(block)['Time'].transform(lambda t: (t - t.min()).dt.total_seconds())
    df['settled'] = held >= SETTLE_SECONDS

    info = dict(files_read=len(frames), files_skipped=skipped,
                rows_raw=n0, duplicates_dropped=n0 - len(df),
                relabelled_files=relabelled,
                compressor_off_rows=int((~df['comp_on']).sum()),
                days=sorted(str(d) for d in df['date'].unique()))
    return df, info


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(p: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=p['n_estimators'], learning_rate=p['learning_rate'],
        max_depth=p['max_depth'], subsample=p['subsample'],
        colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'],
        reg_lambda=p['reg_lambda'], objective='reg:squarederror',
        random_state=SEED, n_jobs=-1)


def leave_one_day_out(X: pd.DataFrame, y: np.ndarray, days: np.ndarray, p: dict) -> dict:
    """The only validation that answers 'what will tomorrow cost at setting X'.

    A random split puts 15-second neighbours on both sides and scores ~0.9
    regardless of whether the model learned anything transferable.
    """
    rows, preds, actual = [], [], []
    for d in sorted(set(days)):
        tr, te = days != d, days == d
        if tr.sum() < 50 or te.sum() < 20:
            continue
        # Use .iloc for row indexing on the pandas DataFrame
        m = make_model(p).fit(X.iloc[tr], y[tr])
        pr = m.predict(X.iloc[te])
        preds.append(pr)
        actual.append(y[te])
        rows.append(dict(day=str(d), n=int(te.sum()),
                         r2=float(r2_score(y[te], pr)),
                         mae=float(mean_absolute_error(y[te], pr)),
                         actual_kw=float(y[te].mean()),
                         predicted_kw=float(pr.mean())))
    if not rows:
        return dict(per_day=[], pooled_r2=float('nan'), pooled_mae=float('nan'),
                    pooled_rmse=float('nan'), days_passed=0, n_days=0)
    preds, actual = np.concatenate(preds), np.concatenate(actual)
    return dict(per_day=rows,
                pooled_r2=float(r2_score(actual, preds)),
                pooled_mae=float(mean_absolute_error(actual, preds)),
                pooled_rmse=float(np.sqrt(mean_squared_error(actual, preds))),
                days_passed=int(sum(r['r2'] > 0 for r in rows)),
                n_days=len(rows))


def random_split_r2(X: pd.DataFrame, y: np.ndarray, p: dict) -> float:
    """Reported only so the metrics show how far it overstates the truth."""
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        pred[te] = make_model(p).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return float(r2_score(y, pred))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_estimators', type=int, default=400)
    ap.add_argument('--learning_rate', type=float, default=0.05)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--subsample', type=float, default=0.8)
    ap.add_argument('--colsample_bytree', type=float, default=0.8)
    ap.add_argument('--min_child_weight', type=float, default=5)
    ap.add_argument('--reg_lambda', type=float, default=1.0)
    ap.add_argument('--model-dir', default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    ap.add_argument('--train', default=os.environ.get('SM_CHANNEL_TRAINING',
                                                      '/opt/ml/input/data/training'))
    args = ap.parse_args()
    p = vars(args)

    df, info = load(args.train)
    # Ensure index is cleanly reset on the training slice
    train = df[df['comp_on'] & df['settled'] & df[TARGET].notna()].copy().reset_index(drop=True)
    features = VALVES + CONDITIONS
    missing = [c for c in features if c not in train.columns]
    if missing:
        sys.exit(f'Missing required columns: {missing}')

    X = train[features].astype(float)
    y = train[TARGET].values
    days = train['date'].values

    log.info('%d files, %d usable rows, %d days',
             info['files_read'], len(train), len(info['days']))

    val = leave_one_day_out(X, y, days, p)
    rnd = random_split_r2(X, y, p)
    pass_rate = val['days_passed'] / val['n_days'] if val['n_days'] else 0.0
    actionable = bool(val['pooled_r2'] > MIN_POOLED_R2 and pass_rate >= MIN_DAY_PASS_RATE)

    # SageMaker scrapes these from the log into training-job metrics
    log.info('validation:pooled_r2=%.4f;', val['pooled_r2'])
    log.info('validation:pooled_mae=%.4f;', val['pooled_mae'])
    log.info('validation:random_split_r2=%.4f;', rnd)
    log.info('validation:days_passed=%d;', val['days_passed'])
    log.info('model actionable: %s', actionable)
    if not actionable:
        log.warning('Model FAILS leave-one-day-out. It will be served in '
                    'advisory mode and the dashboard will show it as unverified.')

    model = make_model(p).fit(X, y)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(out / 'model.json'))

    observed = (train.groupby(VALVES)[TARGET]
                .agg(['mean', 'size', 'count']).reset_index())
    obs_days = train.groupby(VALVES)['date'].nunique().reset_index(name='days')
    observed = observed.merge(obs_days, on=VALVES)

    metadata = dict(
        target=TARGET,
        features=features,
        valves=VALVES,
        conditions=CONDITIONS,
        valve_options={v: sorted(int(x) for x in train[v].unique()) for v in VALVES},
        observed_combinations=[
            dict(valves={v: int(r[v]) for v in VALVES},
                 measured_kw=round(float(r['mean']), 4),
                 rows=int(r['size']), days=int(r['days']))
            for _, r in observed.iterrows()],
        condition_ranges={c: [float(train[c].min()), float(train[c].max())]
                          for c in CONDITIONS},
        condition_medians={c: float(train[c].median()) for c in CONDITIONS},
        feature_importance=dict(zip(features,
                                     [float(v) for v in model.feature_importances_])),
        validation=dict(**val, random_split_r2=rnd, pass_rate=pass_rate,
                        min_pooled_r2=MIN_POOLED_R2,
                        min_day_pass_rate=MIN_DAY_PASS_RATE),
        actionable=actionable,
        data=info,
        hyperparameters={k: p[k] for k in
                         ('n_estimators', 'learning_rate', 'max_depth', 'subsample',
                          'colsample_bytree', 'min_child_weight', 'reg_lambda')},
        training_rows=int(len(train)),
    )
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    log.info('wrote model.json and metadata.json to %s', out)


if __name__ == '__main__':
    main()
