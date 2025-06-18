#!/bin/bash

curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "searching for a document using this API"}'