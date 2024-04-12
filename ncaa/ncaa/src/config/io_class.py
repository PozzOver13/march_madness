from pydantic import BaseModel, Field, field_validator

from typing_extensions import Annotated

from ncaa.src.config.general import TEAM_KNOWN

class Matchup(BaseModel):
    team: Annotated[
        str,
        Field(...,
              title="Team",
              description="The team to compare",
              min_length=1)
    ]
    team_opponent: Annotated[
        str,
        Field(...,
              title="Opponent",
              description="The opponent team to compare",
              min_length=1)
    ]

    @field_validator('team', 'team_opponent')
    @classmethod
    def check_team_allowed(cls, team: str):
        if team not in TEAM_KNOWN:
            raise ValueError("Team or opponent not allowed")
        else:
            return team



class MatchupOutput(BaseModel):
    team: str = Field(...,
                      title="Team",
                      description="The team winning the matchup",
                      min_length=1)
    wins: int = Field(...,
                      title="Wins",
                      description="The number of wins",
                      ge=0)
