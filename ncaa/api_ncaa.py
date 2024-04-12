from fastapi import FastAPI

from ncaa.api_endpoint import get_top_3_wins, get_wins_by_team, get_winner
from ncaa.src.config.io_class import Matchup

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the NCAA API!"}

@app.get("/top_3_wins")
async def top_3_wins():
    return get_top_3_wins()

@app.get("/wins_by_team")
async def wins_by_team(team: str):
    return get_wins_by_team(team)

@app.post("/who_wins")
async def who_wins(matchup: Matchup):
    return get_winner(matchup)