from fastapi import FastAPI

from ncaa.api import get_top_3_wins

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the NCAA API!"}

@app.get("/top_3_wins")
async def top_3_wins():
    return get_top_3_wins()
