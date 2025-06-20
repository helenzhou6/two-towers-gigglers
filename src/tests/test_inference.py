from inference import search_query
from inference_manual import search_query as search_query_manual

# TO TEST, run:
QUERY = "home pickled eggs causing botulism at room temperature"
results = search_query(QUERY)
print("----- REDIS RESULTS ------")

for result in results:
    print(result)

results = search_query_manual(QUERY)
print("----- MANUAL (pre redis) RESULTS ------")
for result in results:
    print(result)