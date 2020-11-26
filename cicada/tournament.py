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

N_ROUNDS = 180
N_PROCESSES = 2


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
