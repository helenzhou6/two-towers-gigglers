from inference import search_query

# TO TEST, run:
QUERY = "home pickled eggs causing botulism at room temperature"
results = search_query(QUERY)
for result in results:
    print(result)