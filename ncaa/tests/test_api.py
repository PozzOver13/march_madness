from api_ncaa import app
from fastapi.testclient import TestClient


def test_root():
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the NCAA API!"}

def test_top_3_wins():
    client = TestClient(app)

    response = client.get("/top_3_wins")
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert response.json()[0]['TEAM'] == 'Gonzaga'
    assert response.json()[1]['TEAM'] == 'Kansas'
    assert response.json()[2]['TEAM'] == 'Duke'

def test_wins_by_team():
    client = TestClient(app)

    response = client.get("/wins_by_team?team=Gonzaga")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]['TEAM'] == 'Gonzaga'
    assert response.json()[0]['wins'] == 307
    assert response.json()[0]['losses'] == 41

def test_wins_by_team_no_team():
    client = TestClient(app)

    response = client.get("/wins_by_team?team=")
    assert response.status_code == 422
    assert response.json() == {"detail": "Team is required"}


def test_who_wins():
    client = TestClient(app)

    response = client.post("/who_wins", json={"team": "Purdue", "team_opponent": "Connecticut"})
    assert response.status_code == 200
    assert response.json() == {"team": "Purdue", "wins": 236}


def test_who_wins_team_integer():
    client = TestClient(app)

    response = client.post("/who_wins", json={"team": 1, "team_opponent": "Connecticut"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid string"


def test_who_wins_team_empty():
    client = TestClient(app)

    response = client.post("/who_wins", json={"team": "", "team_opponent": "Connecticut"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "String should have at least 1 character"