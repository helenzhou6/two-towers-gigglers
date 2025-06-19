#!/bin/sh

export PYTHONPATH=./src

python3 src/index_docs_to_redis.py &

uvicorn src.api:app --host 0.0.0.0 --port 8000
