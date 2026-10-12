"""Lambda 2 - S3 ObjectCreated (EventBridge) -> SageMaker endpoint -> DynamoDB.

Triggered when a PLC record lands in s3://<bucket>/raw/. Reads the
record, pulls the four ambient conditions out of it, asks the endpoint for the
best valve setting, and stores the answer where the dashboard can read it.

Writes two DynamoDB items per invocation:
    PK = "REC", SK = "LATEST"               current recommendation for dashboard
    PK = "REC", SK = "HIST#<iso>#<key>"     time-series history row, TTL'd

Environment
    ENDPOINT_NAME     SageMaker endpoint          (required)
    TABLE_NAME        DynamoDB table              (required)
    SEARCH            "observed" (default) | "full"
    TOP_K             alternatives to keep, default 5
    HISTORY_TTL_DAYS  default 30
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import unquote_plus

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

s3 = boto3.client('s3')
runtime = boto3.client('sagemaker-runtime')
table = boto3.resource('dynamodb').Table(os.environ['TABLE_NAME'])

ENDPOINT = os.environ['ENDPOINT_NAME']
SEARCH = os.environ.get('SEARCH', 'observed')
TOP_K = int(os.environ.get('TOP_K', '5'))
TTL_DAYS = int(os.environ.get('HISTORY_TTL_DAYS', '30'))

WANTED = {
    'outdoor_temp': ['outdoor_temp', 'Outdoor-T8(°C)', 'Outdoor-T8(C)', 'outdoor_t8'],
    'outdoor_humidity': ['outdoor_humidity', 'Outdoor Humidity(%)', 'outdoor_rh'],
    'room_temp': ['room_temp', 'Room-T7(°C)', 'Room-T7(C)', 'room_t7'],
    'indoor_humidity': ['indoor_humidity', 'Indoor Humidity(%)', 'indoor_rh'],
}


def lambda_handler(event, context):
    records = []
    # Support direct S3 notifications or EventBridge wrapped events
    if 'Records' in event:
        for r in event['Records']:
            if 's3' in r and 'bucket' in r['s3']:
                records.append((r['s3']['bucket']['name'], unquote_plus(r['s3']['object']['key'])))
    elif event.get('source') == 'aws.s3':
        detail = event.get('detail', {})
        bucket = detail.get('bucket', {}).get('name')
        key = unquote_plus(detail.get('object', {}).get('key', ''))
        if bucket and key:
            records.append((bucket, key))

    results, failed = [], []
    for bucket, key in records:
        if key.endswith('/'):
            log.info('Skipping directory marker: s3://%s/%s', bucket, key)
            continue
        try:
            results.append(process(bucket, key))
        except Exception:
            log.exception('Failed processing s3://%s/%s', bucket, key)
            failed.append(key)
            results.append({'key': key, 'ok': False})

    # Re-raise if any record failed so async Lambda retries and alarms fire
    if failed:
        raise RuntimeError(f'Failed processing {len(failed)}/{len(records)} record(s): {failed}')

    return {'processed': results}


def process(bucket: str, key: str) -> dict:
    body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    reading = latest_reading(body, key)
    conditions = extract(reading)

    payload = {**conditions, 'search': SEARCH, 'top_k': TOP_K}
    t0 = time.time()
    resp = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                   ContentType='application/json',
                                   Body=json.dumps(payload).encode())
    out = json.loads(resp['Body'].read())
    log.info('Endpoint %s returned %s in %.0f ms',
             ENDPOINT, out.get('status'), (time.time() - t0) * 1000)

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec='seconds')
    item = {
        'pk': 'REC',
        'sk': 'LATEST',
        'updated_at': now_iso,
        'source_key': key,
        'status': out.get('status'),
        'actionable': bool(out.get('actionable')),
        'recommended': out.get('recommended'),
        'alternatives': out.get('alternatives', []),
        'runner_up_gap_kw': out.get('runner_up_gap_kw'),
        'model_error_kw': out.get('model_error_kw'),
        'separable': bool(out.get('separable')),
        'warnings': out.get('warnings', []),
        'conditions': conditions,
        'model_version': out.get('model_version'),
        'generated_at': out.get('generated_at'),
    }
    table.put_item(Item=to_dynamo(item))

    history = dict(item,
                   pk='REC',
                   sk=f"HIST#{now_iso}#{key}",
                   ttl=int(now.timestamp()) + TTL_DAYS * 86400)
    table.put_item(Item=to_dynamo(history))

    return {'key': key, 'ok': True, 'status': out.get('status')}


def latest_reading(body: bytes, key: str) -> dict:
    """Lambda 1 may write a single JSON object, JSON lines, or CSV."""
    try:
        text = body.decode('utf-8-sig').strip()
    except UnicodeDecodeError:
        text = body.decode('latin-1').strip()

    if not text:
        raise ValueError(f'{key} is empty')

    if key.lower().endswith('.csv'):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise ValueError(f'{key} has no CSV rows')
        return rows[-1]

    try:
        data = json.loads(text)
        if isinstance(data, list):
            if not data:
                raise ValueError(f'{key} is an empty list')
            return data[-1]
        if isinstance(data, dict):
            for wrapper in ('record', 'data', 'payload', 'body'):
                if isinstance(data.get(wrapper), dict):
                    return data[wrapper]
            return data
    except json.JSONDecodeError:
        pass

    last = [ln for ln in text.splitlines() if ln.strip()][-1]
    res = json.loads(last)
    if not isinstance(res, dict):
        raise ValueError(f'Expected dict from line in {key}, got {type(res).__name__}')
    return res


def extract(reading: dict) -> dict:
    """Pull the four conditions out, whatever the producer called them."""
    lowered = {str(k).strip().lower(): v for k, v in reading.items() if v is not None and v != ''}
    out, missing = {}, []
    for want, names in WANTED.items():
        value = next((reading[n] for n in names if n in reading and reading[n] is not None and reading[n] != ''), None)
        if value is None:
            value = next((lowered[n.lower()] for n in names if n.lower() in lowered), None)
        if value is None:
            missing.append(want)
            continue
        try:
            val_float = float(value)
            if not math.isfinite(val_float):
                missing.append(want)
            else:
                out[want] = val_float
        except (TypeError, ValueError):
            missing.append(want)
    if missing:
        raise ValueError(f'Record is missing {missing}. Saw keys: {sorted(reading)[:20]}')
    return out


def to_dynamo(obj):
    """DynamoDB stores Decimal, not float. Omit non-finite numbers."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: to_dynamo(v) for k, v in obj.items() if v is not None and to_dynamo(v) is not None}
    if isinstance(obj, list):
        return [to_dynamo(v) for v in obj if v is not None and to_dynamo(v) is not None]
    return obj
