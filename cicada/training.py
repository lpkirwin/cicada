import datetime
from multiprocessing import Pool

from kaggle_environments import make

from cicada import agent
from cicada.utils import data, models

INIT_NEW_FILES = False
N_GAMES_PER_ROUND = 350
N_ROUNDS = 3
N_PROCESSES = 5
NOISE_SD = 0.05


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
            # "episodeSteps": 200,
        },
    )
    env.reset()
    env.run([action, "cicada/simple_rules_based_agent.py"])

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

    record_type_counts = data.count_record_types(filtered_log)

    score = env.state[0]["observation"]["players_raw"][0]["score"]
    data.add_to_score_file(score)

    to_print = f"game_num: {str(game_num):>4}, score: {str(score):>8}, "
    rec_count_strings = list()
    for rec_type in [
        "SHOT_ATTEMPT",
        "MOVE_WITH_BALL_ATTEMPT",
        "SHORT_PASS_ATTEMPT",
        "LONG_PASS_ATTEMPT",
        "LOST_POSSESSION",
        "KICK_WITH_NO_ATTEMPT_EVENT",
    ]:
        rec_count_strings.append(
            f"{rec_type}: {str(record_type_counts.get(rec_type, 0)):>4}"
        )
    to_print += ", ".join(rec_count_strings)
    print(to_print)


if __name__ == "__main__":

    if INIT_NEW_FILES:
        data.init_log_file()
        data.init_score_file()
        n_games_start = 0
        print("initialised new game data")
    else:
        n_games_start = data.get_n_games()
        print(f"adding to existing {n_games_start} games")

    agent.plans.models.load_lgb_models(quiet=False)

    for round in range(N_ROUNDS):
        start_time = datetime.datetime.now()
        print("simulating matches...")

        with Pool(processes=N_PROCESSES) as pool:
            pool.map(simulate_one_game, range(N_GAMES_PER_ROUND))

        end_time = datetime.datetime.now()
        time_diff = end_time - start_time

        print("round", round, "done")
        print("elapsed time:", time_diff)
        print("seconds per game:", time_diff.total_seconds() / N_GAMES_PER_ROUND)
        print("average scores:")
        scores = data.get_score_file_as_df()
        print(scores.iloc[n_games_start:].mean())
        n_games_start += N_GAMES_PER_ROUND

        for name, spec in models.lgb_model_specs.items():
            models.fit_lgb_model(spec)
        agent.plans.models.load_lgb_models(quiet=False)

    print("done all rounds")
