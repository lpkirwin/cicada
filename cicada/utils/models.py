import json
import os

from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from cicada.utils import data
from cicada.utils import config

FILEPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_dataset(
    inner_function,
    filename,
    incremental=config.INCREMENTAL_DATA,
    max_rows=config.MAX_DATASET_ROWS,
):

    df_path = os.path.join(FILEPATH, "data", filename)

    if incremental:
        try:
            existing_df = pd.read_pickle(df_path)
            existing_length = existing_df.game_id.max() + 1
        except FileNotFoundError:
            existing_df = None
            existing_length = 0
    else:
        existing_df = None
        existing_length = 0

    dfs = list()
    game_id = existing_length

    with data.get_log_file() as file:
        for i, log_string in tqdm(enumerate(file), total=data.get_n_games()):

            if i < existing_length:
                continue

            log = json.loads(log_string)
            df = inner_function(log)

            if len(df):
                df["game_id"] = game_id
                dfs.append(df)

            game_id += 1

    if existing_length > 0:
        dfs = [existing_df] + dfs

    new_df = pd.concat(dfs).reset_index(drop=True)

    if len(new_df) > max_rows:
        print("truncating dataset longer than", max_rows, "rows")
        new_df = new_df.tail(max_rows)

    new_df.to_pickle(df_path)

    return new_df


def shot_inner_function(log, n_steps=30):

    filtered_log = data.filter_log(
        log,
        type=("SHOT_ATTEMPT", "GOAL_SCORED"),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["target"] = (
            (df.type == "SHOT_ATTEMPT")
            & (df.type.shift(-1) == "GOAL_SCORED")
            & (df.step.shift(-1) - df.step <= n_steps)
        ).astype(int)
        df = df[df.type == "SHOT_ATTEMPT"]

    return df


def short_pass_inner_function(log, n_steps=30):

    filtered_log = data.filter_log(
        log, type=("SHORT_PASS_ATTEMPT", "NEW_POSSESSION", "OPP_POSSESSION")
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["target"] = (
            (df.type == "SHORT_PASS_ATTEMPT")
            & (df.type.shift(-1) == "NEW_POSSESSION")
            & (df.step.shift(-1) - df.step <= n_steps)
        ).astype(int)
        df = df[df.type == "SHORT_PASS_ATTEMPT"]

    return df


def long_pass_inner_function(log, n_steps=30):

    filtered_log = data.filter_log(
        log,
        type=("LONG_PASS_ATTEMPT", "NEW_POSSESSION", "OPP_POSSESSION"),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["target"] = (
            (df.type == "LONG_PASS_ATTEMPT")
            & (df.type.shift(-1) == "NEW_POSSESSION")
            & (df.step.shift(-1) - df.step <= n_steps)
        ).astype(int)
        df = df[df.type == "LONG_PASS_ATTEMPT"]

    return df


def handle_inner_function(log, n_steps=10):

    filtered_log = data.filter_log(
        log,
        type=("MOVE_WITH_BALL_ATTEMPT", "LOST_POSSESSION", "OPP_POSSESSION"),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["target"] = 1

        lost_poss_steps = df.step[df.type == "LOST_POSSESSION"]
        for step in lost_poss_steps:

            mask = (df.step < step) & (df.step >= (step - n_steps))
            df.loc[mask, "target"] = 0  # failed if lost possession

        df = df[df.type == "MOVE_WITH_BALL_ATTEMPT"]

    return df


def position_score_inner_function(log):

    filtered_log = data.filter_log(
        log,
        type=("ACTIVE_POS_SCORE", "GOAL_SCORED", "OPP_POSSESSION"),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["reward"] = 0

        goal_steps = df.step[df.type == "GOAL_SCORED"]
        for step in goal_steps:
            mask = (df.step < step) & (df.step >= (step - 20))
            df.loc[mask, "reward"] = 1.0

        opp_pos_steps = df.step[df.type == "OPP_POSSESSION"]
        for step in opp_pos_steps:
            mask = (df.step < step) & (df.step >= (step - 10))
            df.loc[mask, "reward"] = -0.01

        df = df[df.type == "ACTIVE_POS_SCORE"]

    return df


make_shot_dataset = partial(
    make_dataset,
    inner_function=shot_inner_function,
    filename="shot_dataset.pkl",
)
make_short_pass_dataset = partial(
    make_dataset,
    inner_function=short_pass_inner_function,
    filename="short_pass_dataset.pkl",
)
make_long_pass_dataset = partial(
    make_dataset,
    inner_function=long_pass_inner_function,
    filename="long_pass_dataset.pkl",
)
make_handle_dataset = partial(
    make_dataset,
    inner_function=handle_inner_function,
    filename="handle_dataset.pkl",
)
make_position_score_dataset = partial(
    make_dataset,
    inner_function=position_score_inner_function,
    filename="position_score_dataset.pkl",
)


lgb_model_specs = {
    "short_pass_success": {
        "filename": "short_pass_model.txt",
        "dataset": make_short_pass_dataset,
        "features": {
            "pass_error_diff": "eval_data.pass_error_diff",
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_dopp": "pos_score_data.dopp",
            "pos_score_kopp": "pos_score_data.kopp",
            "small_cone_angle": "eval_data.small_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
            # "timestep": "eval_data.timestep",
            # "kick_countdown": "eval_data.kick_countdown",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 301, 30)),
            "num_leaves": [5, 30],
        },
        "scoring": "neg_log_loss",
        "default_prediction": 0.8,
    },
    "long_pass_success": {
        "filename": "long_pass_model.txt",
        "dataset": make_long_pass_dataset,
        "features": {
            "pass_error_diff": "eval_data.pass_error_diff",
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_dopp": "pos_score_data.dopp",
            "pos_score_kopp": "pos_score_data.kopp",
            "small_cone_angle": "eval_data.small_cone_angle",
            "forward_cone_angle": "eval_data.forward_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
            # "timestep": "eval_data.timestep",
            # "kick_countdown": "eval_data.kick_countdown",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 301, 30)),
            "num_leaves": [5, 30],
        },
        "scoring": "neg_log_loss",
        "default_prediction": 0.8,
    },
    "handle_success": {
        "filename": "handle_model.txt",
        "dataset": make_handle_dataset,
        "features": {
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_view": "pos_score_data.view",
            "pos_score_dopp": "pos_score_data.dopp",
            "pos_score_kopp": "pos_score_data.kopp",
            "close_opp_dir_change": "eval_data.close_opp_dir_change",
            "small_cone_angle": "eval_data.small_cone_angle",
            "angle_diff": "eval_data.angle_diff",
            # "timestep": "eval_data.timestep",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(50, 551, 100)),
            "num_leaves": [30],
        },
        "scoring": "neg_log_loss",
        "default_prediction": 0.8,
    },
    "shot_success": {
        "filename": "shot_model.txt",
        "dataset": make_shot_dataset,
        "features": {
            "view_of_net": "eval_data.view_of_net",
            "distance_to_net": "eval_data.distance_to_net",
            "distance_to_goalie": "eval_data.dist_to_goalie",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 301, 30)),
            "num_leaves": [10],
        },
        "scoring": "neg_log_loss",
        "default_prediction": 0.1,
    },
}


def fit_lgb_model(model_spec):
    ms = model_spec
    print("loading data using", ms["filename"])
    df = ms["dataset"]()
    print(df.describe().T)
    grid_search = GridSearchCV(
        estimator=ms["class"](),
        param_grid=ms["grid"],
        scoring=ms["scoring"],
        verbose=1,
        n_jobs=5,
    )
    X = df[ms["features"].values()]
    y = df["target"]
    n_games = df["game_id"].max() + 1
    game_pct = (df["game_id"] + 1) / n_games
    weights = (
        config.GAME_WEIGHTING_FACTOR + (1 - config.GAME_WEIGHTING_FACTOR) * game_pct
    )
    print("fitting", ms["filename"])
    grid_search.fit(X, y, sample_weight=weights)
    print("best params:", grid_search.best_params_)
    pred = grid_search.predict_proba(X)[:, 1]
    print("distribution of predictions:")
    print(
        pd.Series(pred).describe(
            percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )
    filepath = os.path.join(FILEPATH, "models", ms["filename"])
    print("saving to", filepath)
    grid_search.best_estimator_.booster_.save_model(filepath)


class PlaceholderModel:
    def __init__(self, value=0.5):
        self.value = value

    def predict(self, *args, **kwargs):
        return [self.value]


def load_lgb_models(quiet=True):

    global lgb_models
    lgb_models = dict()

    for name, spec in lgb_model_specs.items():

        filepath = os.path.join(FILEPATH, "models", spec["filename"])
        if os.path.isfile(filepath):
            model = lgb.Booster(model_file=filepath)
            if not quiet:
                print("model loaded from", filepath)
        else:
            if not quiet:
                print("model not found, using placeholder instead:", name)
            model = PlaceholderModel(spec["default_prediction"])

        lgb_models[name] = model


load_lgb_models()


def make_predict_function(name):
    model_spec = lgb_model_specs[name]
    feature_names = ", ".join(model_spec["features"].keys())
    function_string = f"def _predict_func({feature_names}): return lgb_models['{name}'].predict([[{feature_names}]])[0]"
    exec(function_string)
    return locals()["_predict_func"]


short_pass_success = make_predict_function("short_pass_success")
long_pass_success = make_predict_function("long_pass_success")
handle_success = make_predict_function("handle_success")
shot_success = make_predict_function("shot_success")


def position_score(
    posx,
    posy,
    dnet,
    view,
    dopp,
    kopp,
):
    return max(
        min(
            468.74842421261667
            + -90.41513287612594 * posx
            + 4.095688444021861 * posx * (posx > -0.5)
            + 70.42702891705054 * posx * (posx > 0.0)
            + 12.642795700464502 * posx * (posx > 0.5)
            + -890.6776542323154 * np.log10(posx + 2.0)
            + -14.342713413697245 * abs(posy)
            + 148.7364449664164 * posy ** 2
            + -96.22550776273538 * dnet
            + -87.97648632100884 * dnet ** 2
            + 17.15915753520119 * view
            + -6.994788902368123 * view ** 2
            + 13.657952325903146 * dopp
            + -2.515510877798804 * dopp * (dopp < 0.05)
            + 6.136153757388059 * dopp * (dopp < 0.1)
            + -105.32057391317849 * dopp ** 2
            + -0.008149365889387297 * kopp
            + -3.7405569397024063 * np.log10(kopp),
            50.0,
        ),
        0,
    )


if __name__ == "__main__":

    # test prediction functions
    for name, spec in lgb_model_specs.items():
        print(name)
        features = spec["features"].keys()
        func = make_predict_function(name)
        pred = func(**{k: 1.0 for k in features})
        print(pred)

    print("tests passed")

    print("training models on existing data")
    for name, spec in lgb_model_specs.items():
        fit_lgb_model(spec)
