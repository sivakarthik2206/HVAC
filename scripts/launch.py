"""Train the model in SageMaker and deploy it behind an endpoint.

    python scripts/launch.py --bucket my-plc-bucket --role arn:aws:iam::...:role/...

Both values come from the CloudFormation outputs. Run `deploy.sh` first.

Serverless inference is the default: the endpoint scales to zero between
readings, so a plant logging every 15 s pays for compute it uses rather than
for an instance idling overnight. Pass --realtime for a provisioned instance
if you need single-digit-millisecond latency or very high throughput.
"""
from __future__ import annotations

import argparse
import json
import sys

import boto3
import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.xgboost import XGBoost, XGBoostModel

# Supported SageMaker XGBoost framework versions at time of writing:
# 1.2-1, 1.2-2, 1.3-1, 1.5-1, 1.7-1, 3.0-5. Pin one; never use :latest.
FRAMEWORK_VERSION = '1.7-1'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', required=True, help='DataBucket from the stack outputs')
    ap.add_argument('--role', required=True, help='SageMakerRoleArn from the stack outputs')
    ap.add_argument('--endpoint-name', default='hvac-valve-optimiser')
    ap.add_argument('--train-prefix', default='training/')
    ap.add_argument('--instance-type', default='ml.m5.large',
                    help='training instance')
    ap.add_argument('--realtime', action='store_true',
                    help='provisioned endpoint instead of serverless')
    ap.add_argument('--realtime-instance', default='ml.t2.medium')
    ap.add_argument('--memory-mb', type=int, default=2048,
                    help='serverless endpoint memory')
    ap.add_argument('--max-concurrency', type=int, default=5)
    ap.add_argument('--skip-training', action='store_true',
                    help='redeploy the last trained model without retraining')
    ap.add_argument('--model-data', help='s3://.../model.tar.gz when skipping training')
    args = ap.parse_args()

    session = sagemaker.Session()
    region = session.boto_region_name
    train_uri = f's3://{args.bucket}/{args.train_prefix}'
    print(f'region          {region}')
    print(f'training data   {train_uri}')
    print(f'endpoint        {args.endpoint_name}')

    if args.skip_training:
        if not args.model_data:
            sys.exit('--skip-training needs --model-data s3://.../model.tar.gz')
        model_data = args.model_data
    else:
        if not _has_objects(args.bucket, args.train_prefix):
            sys.exit(f'No objects under {train_uri}. Upload the PLC exports first:\n'
                     f'  aws s3 cp ./plc_data {train_uri} --recursive')

        estimator = XGBoost(
            entry_point='train.py',
            source_dir='src/train',
            framework_version=FRAMEWORK_VERSION,
            py_version='py3',
            role=args.role,
            instance_type=args.instance_type,
            instance_count=1,
            output_path=f's3://{args.bucket}/models/',
            base_job_name='hvac-valve',
            sagemaker_session=session,
            hyperparameters={
                'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'min_child_weight': 5, 'reg_lambda': 1.0,
            },
            metric_definitions=[
                {'Name': 'validation:pooled_r2',
                 'Regex': r'validation:pooled_r2=([0-9.\-]+);'},
                {'Name': 'validation:pooled_mae',
                 'Regex': r'validation:pooled_mae=([0-9.\-]+);'},
                {'Name': 'validation:random_split_r2',
                 'Regex': r'validation:random_split_r2=([0-9.\-]+);'},
            ],
        )
        print('\nstarting training job...')
        estimator.fit({'training': train_uri}, wait=True)
        model_data = estimator.model_data
        print(f'model artifact  {model_data}')

    model = XGBoostModel(
        model_data=model_data,
        role=args.role,
        entry_point='inference.py',
        source_dir='src/inference',
        framework_version=FRAMEWORK_VERSION,
        py_version='py3',
        sagemaker_session=session,
    )

    print('\ndeploying endpoint (a few minutes)...')
    if args.realtime:
        predictor = _update_or_create(
            model=model,
            endpoint_name=args.endpoint_name,
            region=region,
            deploy_kwargs=dict(
                initial_instance_count=1,
                instance_type=args.realtime_instance,
            ),
        )
        print(f'deployed on {args.realtime_instance} (billed continuously)')
    else:
        predictor = _update_or_create(
            model=model,
            endpoint_name=args.endpoint_name,
            region=region,
            deploy_kwargs=dict(
                serverless_inference_config=ServerlessInferenceConfig(
                    memory_size_in_mb=args.memory_mb,
                    max_concurrency=args.max_concurrency,
                ),
            ),
        )
        print(f'deployed serverless ({args.memory_mb} MB, '
              f'max concurrency {args.max_concurrency}); scales to zero when idle')

    print('\nsmoke test...')
    body = json.dumps({'outdoor_temp': 34.5, 'outdoor_humidity': 55.0,
                       'room_temp': 22.0, 'indoor_humidity': 48.0,
                       'search': 'observed', 'top_k': 3})
    resp = boto3.client('sagemaker-runtime', region_name=region).invoke_endpoint(
        EndpointName=args.endpoint_name, ContentType='application/json',
        Body=body.encode())
    out = json.loads(resp['Body'].read())
    print(json.dumps(out, indent=2)[:1200])

    print(f"\nstatus: {out.get('status')}")
    if not out.get('actionable'):
        print('The model is serving in ADVISORY mode: it did not pass '
              'leave-one-day-out validation, so the dashboard will label the '
              'set points as unverified. See the README for what fixes this.')
    print(f'\nendpoint {args.endpoint_name} is live.')


def _has_objects(bucket: str, prefix: str) -> bool:
    resp = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return resp.get('KeyCount', 0) > 0


def _update_or_create(model, endpoint_name: str, region: str, deploy_kwargs: dict):
    """Deploy or update-in-place with zero downtime.

    Calls model.deploy() which, when the named endpoint already exists, issues
    an UpdateEndpoint to atomically swap the config. If the endpoint is new it
    falls through to a normal CreateEndpoint. Either way we wait for InService.
    """
    sm = boto3.client('sagemaker', region_name=region)
    endpoint_exists = True
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
    except sm.exceptions.ClientError:
        endpoint_exists = False

    if endpoint_exists:
        print(f'updating existing endpoint {endpoint_name} (zero downtime)...')

    # The SageMaker Python SDK calls UpdateEndpoint automatically when the
    # endpoint already exists and we pass the same name.
    predictor = model.deploy(endpoint_name=endpoint_name, **deploy_kwargs)

    # Wait explicitly so we get a clear success/failure message here.
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name,
                WaiterConfig={'Delay': 15, 'MaxAttempts': 80})
    return predictor


if __name__ == '__main__':
    main()
