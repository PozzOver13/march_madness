import os
import polars as pl

from ncaa.src.config.io_class import Matchup, MatchupOutput
from ncaa.src.error_handler import check_team_string_not_empty
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
    check_team_string_not_empty(team)

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


def get_winner(item: Matchup):
    loader = Loader()
    path_file = os.path.join(PATH_DATA_RAW, 'kaggle', 'cbb.csv')
    df = loader.load_data(path_file)

    team_wins = (
        df
        .filter(pl.col('TEAM') == item.team)
        .group_by('TEAM')
        .agg(
            wins=pl.sum('W'),
            wins_pct=pl.sum('W') / pl.sum('G'),
            losses=pl.sum('G') - pl.sum('W'),
        )
        .to_pandas()
    )

    team_opp_wins = (
        df
        .filter(pl.col('TEAM') == item.team_opponent)
        .group_by('TEAM')
        .agg(
            wins=pl.sum('W'),
            wins_pct=pl.sum('W') / pl.sum('G'),
            losses=pl.sum('G') - pl.sum('W'),
        )

        .to_pandas()
    )


    if team_wins['wins'].values[0] > team_opp_wins['wins'].values[0]:
        res = {"team": item.team, "wins": team_wins['wins'].values[0]}
    else:
        res = {"team": item.team_opponent, "wins": team_opp_wins['wins'].values[0]}

    return MatchupOutput(**res)