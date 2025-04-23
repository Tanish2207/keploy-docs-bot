from fastapi import FastAPI, Query
from pydantic import BaseModel
from brain import build_vector_db, answer_query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embed")
def embed_docs():
    build_vector_db()
    return {"status": "Vector DB built/refreshed."}

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/query")
def query_docs(request: QueryRequest):
    result = answer_query(request.query, k=request.k)
    return result
