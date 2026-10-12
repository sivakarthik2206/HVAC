"""Dashboard read API.

    GET /recommendation          the current valve set points
    GET /history?hours=6         recent recommendations for a trend

Returns float, not Decimal, and sets CORS so a browser dashboard can call it.

Environment
    TABLE_NAME     DynamoDB table   (required)
    CORS_ORIGIN    allowed origin, default "*"
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

log = logging.getLogger()
log.setLevel(logging.INFO)
table = boto3.resource('dynamodb').Table(os.environ['TABLE_NAME'])
CORS = os.environ.get('CORS_ORIGIN', '*')
MAX_HOURS = 168


def lambda_handler(event, context):
    raw_path = (event.get('rawPath')
                or event.get('path')
                or event.get('requestContext', {}).get('http', {}).get('path', '/'))
    path = raw_path.rstrip('/')

    method = event.get('requestContext', {}).get('http', {}).get('method', 'GET').upper()
    if method not in ('GET', 'OPTIONS'):
        return reply(405, {'error': f'Method {method} not allowed'})

    if method == 'OPTIONS':
        return reply(204, {})

    params = event.get('queryStringParameters') or {}
    try:
        if path.endswith('/history'):
            return reply(200, history(params))
        elif path.endswith('/recommendation') or path.endswith('/v1'):
            return reply(200, latest())
        else:
            return reply(404, {'error': f'Route not found: {raw_path}'})
    except ValueError as exc:
        return reply(400, {'error': str(exc)})
    except Exception:                                          # noqa: BLE001
        log.exception('Dashboard API failure')
        return reply(500, {'error': 'Internal server error'})


def latest() -> dict:
    item = table.get_item(Key={'pk': 'REC', 'sk': 'LATEST'}).get('Item')
    if not item:
        return {'status': 'NO_DATA',
                'message': 'No recommendation yet. Waiting for the first PLC '
                           'record to land in S3.'}
    return undecimal(item)


def history(params: dict) -> dict:
    try:
        hours = int(params.get('hours', 6))
    except (TypeError, ValueError) as exc:
        raise ValueError("'hours' must be a whole number") from exc
    if not 1 <= hours <= MAX_HOURS:
        raise ValueError(f"'hours' must be between 1 and {MAX_HOURS}")

    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec='seconds')
    query_kwargs = {
        'KeyConditionExpression': Key('pk').eq('REC') & Key('sk').between(f'HIST#{since}', 'HIST#9999'),
        'ScanIndexForward': True
    }

    items = []
    while True:
        resp = table.query(**query_kwargs)
        items.extend(resp.get('Items', []))
        if 'LastEvaluatedKey' in resp:
            query_kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
        else:
            break

    deserialized = [undecimal(i) for i in items]
    return {'hours': hours, 'count': len(deserialized), 'items': deserialized}


def undecimal(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: undecimal(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [undecimal(v) for v in obj]
    return obj


def reply(status: int, body: dict) -> dict:
    return {
        'statusCode': status,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': CORS,
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Cache-Control': 'no-store',
        },
        'body': json.dumps(body) if body or status != 204 else '',
    }
