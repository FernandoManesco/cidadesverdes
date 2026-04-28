from fastapi import FastAPI
from app.database import lifespan

app = FastAPI(
    title="Cidades Verdes API",
    description="Sistema de gestão - Cidades Verdes",
    version="1.5.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"status": "ok", "projeto": "Cidades Verdes"}
