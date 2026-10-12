# HVAC valve optimiser on SageMaker

Lambda 1 writes a PLC record to S3. An S3 event wakes Lambda 2, which asks a
SageMaker endpoint for the valve positions with the lowest predicted power and
stores the answer in DynamoDB. The dashboard reads it over an HTTP API.

```
 PLC ──► Lambda 1 ──► S3  s3://bucket/raw/…
                       │
                       │ ObjectCreated event
                       ▼
                   Lambda 2 ──► SageMaker endpoint  (model + valve search)
                       │              ▲
                       │              │ model.tar.gz
                       ▼              │
                   DynamoDB      S3  models/   ◄── SageMaker training job
                       │                            reads  s3://bucket/training/
                       ▼
             API Gateway ──► Dashboard
             GET /recommendation
             GET /history?hours=6
```

## Layout

| Path | What it is |
|---|---|
| `src/train/train.py` | Training job: loads exports, repairs the data faults, validates leave-one-day-out, writes `model.json` + `metadata.json` |
| `src/inference/inference.py` | Endpoint handler: one record in, ranked valve set points out |
| `src/lambda_infer/app.py` | Lambda 2: S3 event → endpoint → DynamoDB |
| `src/lambda_api/app.py` | Dashboard read API |
| `infra/template.yaml` | SAM stack: bucket, DynamoDB, both Lambdas, HTTP API, IAM, alarm |
| `scripts/launch.py` | Runs the training job and deploys the endpoint |
| `deploy.sh` | Builds and deploys the stack |

## Deploy

Needs AWS CLI, SAM CLI, Python 3.12, and `pip install sagemaker boto3`.

```bash
./deploy.sh my-unique-plc-bucket

# upload historical exports, then train and deploy
aws s3 cp ./plc_data s3://my-unique-plc-bucket/training/ --recursive
python scripts/launch.py --bucket my-unique-plc-bucket --role <SageMakerRoleArn>
```

`deploy.sh` prints the role ARN and the dashboard URL. The endpoint is
**serverless** by default — it scales to zero between readings. Pass
`--realtime` for a provisioned instance.

The bucket name is a parameter rather than generated, because an S3
notification pointing at a Lambda in the same stack deadlocks unless the name
is known before the bucket exists.

## What Lambda 1 has to write

Anything under `raw/`. Lambda 2 accepts a single JSON object, a JSON array,
JSON lines, or CSV, and takes the most recent row. Keys may be friendly names
or the raw PLC column names:

```json
{"outdoor_temp": 34.5, "outdoor_humidity": 55.0,
 "room_temp": 22.0, "indoor_humidity": 48.0}
```

```json
{"Outdoor-T8(°C)": 34.5, "Outdoor Humidity(%)": 55.0,
 "Room-T7(°C)": 22.0, "Indoor Humidity(%)": 48.0}
```

Those four fields are required; everything else in the record is ignored.
Write to `raw/` for live recommendations and `training/` for the history the
next training run learns from. Keep writing full exports to `training/` — the
model improves only with more data.

## The dashboard contract

`GET /recommendation`

```json
{
  "status": "ADVISORY",
  "actionable": false,
  "recommended": {
    "PCV-1(%)": 100, "PCV-2(%)": 95, "PCV-3(%)": 95, "PCV-4(%)": 90,
    "predicted_kw": 3.6769, "measured_kw": 3.6662,
    "rows": 680, "days": 1, "observed": true
  },
  "alternatives": [ … ],
  "runner_up_gap_kw": 0.0223,
  "model_error_kw": 0.0631,
  "separable": false,
  "warnings": [ … ],
  "conditions": { … },
  "model_version": "2026-10-06",
  "generated_at": "2026-10-09T00:57:09+00:00"
}
```

**Build the dashboard around `status`, not around the four numbers.**

| `status` | What the dashboard should show |
|---|---|
| `ACTIONABLE` | Set points, presented as a recommendation |
| `ADVISORY` | The same numbers, labelled unverified, with `warnings` visible |
| `NO_DATA` | Waiting for the first record |

Two fields decide whether a recommendation means anything:

- **`separable`** — false when the top two candidates are closer together than
  the model's own out-of-day error. The ranking cannot tell them apart, so
  showing a single winner would overstate what the model knows.
- **`observed`** on each candidate — false when that valve combination has
  never actually been run. Tree models do not extrapolate; outside their
  training states they return the nearest leaf, so an unobserved candidate's
  prediction carries no information. Grey these out.

`GET /history?hours=6` returns the recent recommendations for a trend line.

## Current model status

Trained on the 28 exports, the model comes back **ADVISORY**:

```
leave-one-day-out pooled R²   +0.119      (needs > 0.50)
days beating their baseline    2 of 7
random 5-fold R²              +0.910      ← what a shuffled split reports
out-of-day MAE                 0.063 kW
runner-up gap                  0.022 kW   ← smaller than the error
```

The random-split number is in there deliberately, so nobody is tempted to
quote it. Rows are 15 s apart and strongly autocorrelated, so shuffling puts
near-identical rows on both sides of the split and scores ~0.9 whether or not
the model learned anything transferable.

The cause is in the data, not the code: each valve combination was run on its
own day, so the valve columns work as day labels and the model leans on them
instead of on weather. The training job computes this itself and sets
`actionable: false` in the artifact; the endpoint reads that flag, and the
status flows through to the dashboard. A model that fails cannot quietly start
issuing set points.

**What flips it to ACTIONABLE:** run two valve settings within the same day,
alternating A-B-A, then repeat that pair on a day with different weather. Four
holds of 15 minutes across two days gives the model its first real contrast.
Upload the new exports to `training/`, re-run `scripts/launch.py`, and watch
`validation:pooled_r2` in the training job's CloudWatch metrics.

## Retraining

`scripts/launch.py` is re-runnable; it replaces the endpoint in place. To put
it on a schedule, add an EventBridge rule that starts a SageMaker Pipeline, or
call the script from a small CodeBuild project. Keep the validation gate in
either path — it is the thing that stops a bad model reaching the plant.

## Running costs

Rough monthly order of magnitude for one plant logging every 15 s:

| Component | Note |
|---|---|
| Serverless endpoint | Billed per request and duration; scales to zero |
| Lambda 2 | ~5,800 invocations/day, well inside the free tier's reach |
| DynamoDB on-demand | Two writes per reading |
| S3 | Raw records expire after a year, old versions after 30 days |
| Training job | Minutes of `ml.m5.large` per run |

A provisioned `ml.t2.medium` endpoint bills continuously whether or not
readings arrive, which is why serverless is the default.

One thing worth weighing: the model is small and the search is over 10 to 96
candidates. Lambda alone could do both, which would remove the endpoint
entirely. SageMaker earns its place here if you want managed retraining, model
versioning and the training metrics; if you only need the arithmetic, the
simpler build is a single Lambda.

## Pinned versions

The SageMaker XGBoost framework container is pinned to `1.7-1` in
`scripts/launch.py`. Supported versions are 1.2-1, 1.2-2, 1.3-1, 1.5-1, 1.7-1
and 3.0-5. Never use `:latest`. Check the current list before upgrading:
<https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>
