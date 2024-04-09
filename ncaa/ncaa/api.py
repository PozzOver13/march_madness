import os
import polars as pl
from fastapi import HTTPException

from ncaa.src.loader import Loader
from ncaa.src.config.general import PATH_DATA_RAW

def get_top_3_wins():
    loader = Loader()
    path_file = os.path.join(PATH_DATA_RAW, 'kaggle', 'cbb.csv')
    df = loader.load_data(path_file)

    team_wins = (
        df
        .group_by('TEAM')
        .agg(
            min_year=pl.min('YEAR'),
            max_year=pl.max('YEAR'),
            wins=pl.sum('W'),
            wins_pct=pl.sum('W') / pl.sum('G'),
            losses=pl.sum('G') - pl.sum('W'),
        )
        .sort('wins', descending=True)
        .head(3)
        .to_pandas()
        .to_dict(orient='records')

    )

    return team_wins


def get_wins_by_team(team: str):
    if team == '':
        raise HTTPException(status_code=400, detail="Team is required")

    loader = Loader()
    path_file = os.path.join(PATH_DATA_RAW, 'kaggle', 'cbb.csv')
    df = loader.load_data(path_file)

    team_wins = (
        df
        .filter(pl.col('TEAM') == team)
        .group_by('TEAM')
        .agg(
            min_year=pl.min('YEAR'),
            max_year=pl.max('YEAR'),
            wins=pl.sum('W'),
            wins_pct=pl.sum('W') / pl.sum('G'),
            losses=pl.sum('G') - pl.sum('W'),
        )
        .to_pandas()
        .to_dict(orient='records')
    )

    return team_wins