from inference import search_query
from inference_manual import search_query as search_query_manual

# TO TEST, run:
# `export REDIS_HOST=localhost` & `export PYTHONPATH=./src`
QUERY = "home pickled eggs causing botulism at room temperature"
NUM_DOCS = 3

print("----- REDIS RESULTS ------")
results = search_query(QUERY, NUM_DOCS)
for result in results:
    print(result)

print("----- MANUAL (pre redis) RESULTS ------")
results = search_query_manual(QUERY, NUM_DOCS)
for result in results:
    print(result)