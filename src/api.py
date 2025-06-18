from fastapi import FastAPI
from pydantic import BaseModel
from inference import search_query

app = FastAPI()

# To run: uvicorn src.api:app --reload
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    return search_query(request.query)

@app.get("/health-check")
def healthcheck():
    return "API up and working"