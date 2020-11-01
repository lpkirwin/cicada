import os
import json
import datetime
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

import pandas as pd
from kaggle_environments import make

from cicada.agent import Agent
from cicada.utils import visualisation as viz

INIT_NEW_FILES = False
N_GAMES_TO_SIMULATE = 10
N_PROCESSES = 5

FILEPATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(FILEPATH, "log.jsonl")
SCORE_PATH = os.path.join(FILEPATH, "score.csv")


# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def init_log_file():
    with open(LOG_PATH, "w") as file:
        file.write("")


def add_to_log_file(obj):

    try:
        json_string = json.dumps(obj, cls=NumpyEncoder)
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


def make_kick_dataset(attempt_event, success_event, n_steps=30):

    dfs = list()
    game_id = 0

    with get_log_file() as file:
        for log in tqdm(file, total=get_n_games()):

            filtered_log = viz.filter_log(
                log=json.loads(log),
                type=(attempt_event, success_event),
            )

            df = viz.parse_log_to_df(filtered_log)

            if len(df):

                df["success"] = (
                    (df.type == attempt_event)
                    & (df.type.shift(-1) == success_event)
                    & (df.step.shift(-1) - df.step <= n_steps)
                ).astype(int)
                df = df[df.type == attempt_event]

                df["game_id"] = game_id
                game_id += 1

                dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def make_handling_dataset(n_steps=10):

    dfs = list()
    game_id = 0

    with get_log_file() as file:
        for log in tqdm(file, total=get_n_games()):

            filtered_log = viz.filter_log(
                log=json.loads(log),
                type=("MOVE_WITH_BALL_ATTEMPT", "LOST_POSSESSION"),
            )

            df = viz.parse_log_to_df(filtered_log)

            if len(df):

                df["failure"] = 0

                lost_poss_steps = df.step[df.type == "LOST_POSSESSION"]
                for step in lost_poss_steps:

                    mask = (df.step < step) & (df.step >= (step - n_steps))
                    df.loc[mask, "failure"] = 1

                df = df[df.type == "MOVE_WITH_BALL_ATTEMPT"]

                df["game_id"] = game_id
                game_id += 1

                dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def make_position_score_dataset(n_steps=20):

    dfs = list()
    game_id = 0

    with get_log_file() as file:
        for log in tqdm(file, total=get_n_games()):

            filtered_log = viz.filter_log(
                log=json.loads(log),
                type=("ACTIVE_POS_SCORE", "GOAL_SCORED"),
            )

            df = viz.parse_log_to_df(filtered_log)

            if len(df):

                df["reward"] = 0

                goal_steps = df.step[df.type == "GOAL_SCORED"]
                for step in goal_steps:

                    mask = (df.step < step) & (df.step >= (step - n_steps))
                    df.loc[mask, "reward"] = 1

                df = df[df.type == "ACTIVE_POS_SCORE"]

                df["game_id"] = game_id
                game_id += 1

                dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def simulate_one_game(game_num):

    agent = Agent()

    def action(obs):
        return agent.action_wrapper(obs)

    env = make(
        environment="football",
        configuration={
            "save_video": False,
            "scenario_name": "11_vs_11_kaggle",
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
    filtered_log = viz.filter_log(agent.state.log, type=log_types_to_keep)
    add_to_log_file(filtered_log)


if __name__ == "__main__":

    if INIT_NEW_FILES:
        init_log_file()
        init_score_file()

    start_time = datetime.datetime.now()

    # simulate_one_game(0)

    with Pool(processes=N_PROCESSES) as pool:
        pool.map(simulate_one_game, range(N_GAMES_TO_SIMULATE))

    end_time = datetime.datetime.now()
    time_diff = end_time - start_time

    print("done")
    print("elapsed time:", time_diff)
    print("seconds per game:", time_diff.total_seconds() / N_GAMES_TO_SIMULATE)
    print("average scores:")

    scores = get_score_file_as_df()
    print(scores.mean())
