import os
import json
import datetime
from multiprocessing import Pool
import importlib

import pandas as pd
from kaggle_environments import make

from cicada import agent
from cicada.utils import models
from cicada.utils import data

INIT_NEW_FILES = True
N_GAMES_PER_ROUND = 5
N_ROUNDS = 4
N_PROCESSES = 5
NOISE_SD = 1.0

FILEPATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(FILEPATH, "log.jsonl")
SCORE_PATH = os.path.join(FILEPATH, "score.csv")


def init_log_file():
    with open(LOG_PATH, "w") as file:
        file.write("")


def add_to_log_file(obj):

    try:
        json_string = json.dumps(obj, cls=data.NumpyEncoder)
    except TypeError as e:
        print("Failed to serialize:", obj)
        raise e

    with open(LOG_PATH, "a") as file:
        file.write(json_string + "\n")


def get_log_file():
    return open(LOG_PATH, "r")


def init_score_file():
    with open(SCORE_PATH, "w") as file:
        file.write("left_score,right_score,win,tie\n")


def add_to_score_file(score):
    with open(SCORE_PATH, "a") as file:
        win = 1 if score[0] > score[1] else 0
        tie = 1 if score[0] == score[1] else 0
        file.write(f"{score[0]},{score[1]},{win},{tie}\n")


def get_score_file_as_df():
    return pd.read_csv(SCORE_PATH)


def get_n_games():
    return len(get_score_file_as_df())


def simulate_one_game(game_num):

    agent_obj = agent.Agent(noise_sd=NOISE_SD)

    def action(obs):
        return agent_obj.action_wrapper(obs)

    env = make(
        environment="football",
        configuration={
            "save_video": False,
            "scenario_name": "11_vs_11_kaggle",
            # "scenario_name": "academy_run_to_score_with_keeper",
            # "episodeSteps": 500,
        },
    )
    env.reset()
    env.run([action, "cicada/submission2.py"])

    score = env.state[0]["observation"]["players_raw"][0]["score"]
    add_to_score_file(score)
    print(f"game_num: {game_num}, score: {score}")

    log_types_to_keep = [
        "SHOT_ATTEMPT",
        "GOAL_SCORED",
        "MOVE_WITH_BALL_ATTEMPT",
        "SHORT_PASS_ATTEMPT",
        "LONG_PASS_ATTEMPT",
        "LOST_POSSESSION",
        "NEW_POSSESSION",
        "ACTIVE_POS_SCORE",
    ]
    filtered_log = data.filter_log(agent_obj.state.log, type=log_types_to_keep)
    add_to_log_file(filtered_log)


if __name__ == "__main__":

    if INIT_NEW_FILES:
        init_log_file()
        init_score_file()
        n_games_start = 0
        print("initialising new game data")
    else:
        n_games_start = get_n_games()
        print(f"adding to existing {n_games_start} games")

    for round in range(N_ROUNDS):

        importlib.reload(agent)

        start_time = datetime.datetime.now()

        with Pool(processes=N_PROCESSES) as pool:
            pool.map(simulate_one_game, range(N_GAMES_PER_ROUND))

        end_time = datetime.datetime.now()
        time_diff = end_time - start_time

        print("round", round, "done")
        print("elapsed time:", time_diff)
        print("seconds per game:", time_diff.total_seconds() / N_GAMES_PER_ROUND)
        print("average scores:")
        scores = get_score_file_as_df()
        print(scores.iloc[n_games_start:].mean())

        for name, spec in models.lgb_model_specs.items():
            models.fit_lgb_model(spec)

    print("done all rounds")
