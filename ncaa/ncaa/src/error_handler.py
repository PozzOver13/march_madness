from fastapi import HTTPException


def check_team_string_not_empty(team):
    if team == '':
        raise HTTPException(status_code=422, detail="Team is required")