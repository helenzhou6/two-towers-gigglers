from redis import Redis
from redis.commands.search.query import Query

import numpy as np

r = Redis()

for i in range(1,4):
    vec = np.random.rand(300).astype(np.float32).tobytes()
    r.hset(f"doc:{i}", mapping={
        "title": f"item {i}",
        "embedding": vec
    })



q = Query(f'*=>[KNN 5 @embedding $vec AS score]').return_fields('score', 'title').dialect(2).sort_by('score')

all_q = (
    Query("*")              # match all docs
      .return_fields("title", "score")  # or list the fields you want
      .dialect(2)           # Redis 7+ requires dialect 2
)


res = r.ft("myIdx").search(q, query_params={"vec": vec})

res_all = r.ft("myIdx").search(all_q)


print(f"found {res.total} results {res_all.total}")
for doc in res.docs:
    print(doc.title, float(doc.score))
