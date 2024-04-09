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
    assert response.status_code == 400
    assert response.json() == {"detail": "Team is required"}

