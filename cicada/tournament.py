import datetime
import importlib.util
import os
import sys
import tarfile
import tempfile
from multiprocessing import Pool
from random import shuffle

import numpy as np
import pandas as pd
from kaggle_environments import make

from cicada.utils import data

N_ROUNDS = 120
N_PROCESSES = 4
GREENLIST_TO_SAVE = ["cicada_202011272336", "cicada_202011272338"]


def find_agent_names():
    files = os.listdir("./submissions")
    return [f.split(".")[0] for f in files if f.endswith(".tar.gz")]


def get_agent_class(agent_name):

    with tempfile.TemporaryDirectory() as tmpdir:

        tarball = tarfile.open(f"./submissions/{agent_name}.tar.gz")
        tarball.extractall(tmpdir)
        tarball.close()

        sys.path.insert(0, tmpdir)

        path_to_module = os.path.join(tmpdir, agent_name, "agent.py")
        module_name = f"{agent_name}.agent"
        spec = importlib.util.spec_from_file_location(module_name, path_to_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        sys.path.pop(0)

    return module.Agent


def simulate_one_game(agent_names):

    agent_0 = get_agent_class(agent_names[0])()
    agent_1 = get_agent_class(agent_names[1])()

    def action_0(obs):
        return agent_0.action_wrapper(obs)

    def action_1(obs):
        return agent_1.action_wrapper(obs)

    env = make(
        environment="football",
        configuration={
            "save_video": False,
            "scenario_name": "11_vs_11_kaggle",
            # "episodeSteps": 10,
        },
    )
    env.reset()
    env.run([action_0, action_1])
    score = env.state[0]["observation"]["players_raw"][0]["score"]

    if agent_names[0] in GREENLIST_TO_SAVE:
        save_score_and_log(agent_0, score)
    if agent_names[1] in GREENLIST_TO_SAVE:
        save_score_and_log(agent_1, score[::-1])

    return score


def report_results(tournament_scores):

    score_df = pd.DataFrame.from_records(tournament_scores)
    print()
    print(score_df.describe())
    print()
    print(score_df.mean())

    win_df = score_df.copy()
    win_df[win_df.columns] = score_df.values == score_df.max(axis=1).values.reshape(
        -1, 1
    )
    win_df[win_df.columns] = win_df.values / win_df.sum(axis=1).values.reshape(-1, 1)
    win_df[score_df.isna()] = np.nan
    print()
    print(win_df.describe())
    print()
    print(win_df.mean())


def save_score_and_log(agent_obj, score):

    log_types_to_keep = [
        "SHOT_ATTEMPT",
        "GOAL_SCORED",
        "OPP_GOAL_SCORED",
        "MOVE_WITH_BALL_ATTEMPT",
        "SHORT_PASS_ATTEMPT",
        "LONG_PASS_ATTEMPT",
        "LOST_POSSESSION",
        "NEW_POSSESSION",
        "OPP_POSSESSION",
        "ACTIVE_POS_SCORE",
        "KICK_WITH_NO_ATTEMPT_EVENT",
        "KICK_RELEASE",
    ]
    filtered_log = data.filter_log(agent_obj.state.log, type=log_types_to_keep)
    data.add_to_log_file(filtered_log)
    data.add_to_score_file(score)


if __name__ == "__main__":

    agent_names = find_agent_names()
    n_matchups = len(agent_names) // 2
    tournament_scores = list()

    for round in range(N_ROUNDS):

        shuffle(agent_names)
        matchups = [
            (agent_names[2 * i], agent_names[2 * i + 1]) for i in range(n_matchups)
        ]

        start_time = datetime.datetime.now()

        print(f"round {round}")
        print(f"simulating {len(matchups)} matchups between {len(agent_names)} agents")
        print(f"start time: {start_time}")

        with Pool(processes=N_PROCESSES) as pool:
            scores = pool.map(simulate_one_game, matchups)

        end_time = datetime.datetime.now()
        time_diff = end_time - start_time

        print("elapsed time:", time_diff)
        print("seconds per game:", time_diff.total_seconds() / len(matchups))

        for matchup, score in zip(matchups, scores):
            rec = {matchup[0]: score[0], matchup[1]: score[1]}
            print(rec)
            tournament_scores.append(rec)

        report_results(tournament_scores)
