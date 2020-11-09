import json
import os

# import joblib
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from cicada.utils import data

FILEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SHORT_PASS_MODEL_PATH = os.path.join(FILEPATH, "short_pass_model.pkl")
# LONG_PASS_MODEL_PATH = os.path.join(FILEPATH, "long_pass_model.pkl")
# HANDLE_MODEL_PATH = os.path.join(FILEPATH, "handle_model.pkl")
# short_pass_model = joblib.load(SHORT_PASS_MODEL_PATH)
# long_pass_model = joblib.load(LONG_PASS_MODEL_PATH)
# handle_model = joblib.load(HANDLE_MODEL_PATH)

POSITION_MODEL_PATH = os.path.join(FILEPATH, "position_model.txt")
position_model = lgb.Booster(model_file=POSITION_MODEL_PATH)


def make_dataset(inner_function, filename, incremental=True):

    df_path = os.path.join(FILEPATH, filename)

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
        for i, log in tqdm(enumerate(file), total=data.get_n_games()):

            if i < existing_length:
                continue

            df = inner_function(log)

            if len(df):
                df["game_id"] = game_id
                dfs.append(df)

            game_id += 1

    if existing_length > 0:
        dfs = [existing_df] + dfs

    new_df = pd.concat(dfs).reset_index(drop=True)
    new_df.to_pickle(df_path)

    return new_df


def shot_inner_function(log, n_steps=30):

    filtered_log = data.filter_log(
        log=json.loads(log),
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
        log=json.loads(log),
        type=("SHORT_PASS_ATTEMPT", "NEW_POSSESSION"),
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
        log=json.loads(log),
        type=("LONG_PASS_ATTEMPT", "NEW_POSSESSION"),
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
        log=json.loads(log),
        type=("MOVE_WITH_BALL_ATTEMPT", "LOST_POSSESSION"),
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


def position_score_inner_function(log, n_steps=20):

    filtered_log = data.filter_log(
        log=json.loads(log),
        type=("ACTIVE_POS_SCORE", "GOAL_SCORED"),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["reward"] = 0

        goal_steps = df.step[df.type == "GOAL_SCORED"]
        for step in goal_steps:

            mask = (df.step < step) & (df.step >= (step - n_steps))
            df.loc[mask, "reward"] = 1

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
            "small_cone_angle": "eval_data.small_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 201, 20)),
            "num_leaves": [10, 30],
        },
        "scoring": "neg_log_loss",
    },
    "long_pass_success": {
        "filename": "long_pass_model.txt",
        "dataset": make_long_pass_dataset,
        "features": {
            "pass_error_diff": "eval_data.pass_error_diff",
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_dopp": "pos_score_data.dopp",
            "small_cone_angle": "eval_data.small_cone_angle",
            "forward_cone_angle": "eval_data.forward_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 201, 20)),
            "num_leaves": [10, 30],
        },
        "scoring": "neg_log_loss",
    },
    "handle_success": {
        "filename": "handle_model.txt",
        "dataset": make_handle_dataset,
        "features": {
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_view": "pos_score_data.view",
            "pos_score_dopp": "pos_score_data.dopp",
            "close_opp_dir_change": "eval_data.close_opp_dir_change",
            "small_cone_angle": "eval_data.small_cone_angle",
            "angle_diff": "eval_data.angle_diff",
        },
        "class": lgb.LGBMClassifier,
        "grid": {
            "n_estimators": list(range(10, 201, 20)),
            "num_leaves": [10, 30],
        },
        "scoring": "neg_log_loss",
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
    print("fitting", ms["filename"])
    grid_search.fit(X, y)
    print("best params:", grid_search.best_params_)
    pred = grid_search.predict_proba(X)[:, 1]
    print("distribution of predictions:")
    print(pd.Series(pred).describe())
    filepath = os.path.join(FILEPATH, ms["filename"])
    print("saving to", filepath)
    grid_search.best_estimator_.booster_.save_model(filepath)


class PlaceholderModel:
    def predict(*args, **kwargs):
        return [0.5]


def hash_of_text_file(filepath):
    try:
        with open(filepath, "r") as file:
            return hash(file.read())
    except FileNotFoundError:
        return 0


def load_lgb_models(quiet=True):

    global lgb_models
    global lgb_model_hashes
    lgb_models = dict()
    lgb_model_hashes = dict()

    for name, spec in lgb_model_specs.items():

        filepath = os.path.join(FILEPATH, spec["filename"])
        if os.path.isfile(filepath):
            model = lgb.Booster(model_file=filepath)
            if not quiet:
                print("model loaded from", filepath)
            model_hash = hash_of_text_file(filepath)
        else:
            if not quiet:
                print("model not found, using placeholder instead:", name)
            model = PlaceholderModel()
            model_hash = 0

        lgb_models[name] = model
        lgb_model_hashes[name] = model_hash


def reload_lgb_models_if_needed(quiet=True):
    if "lgb_models" not in globals():
        return load_lgb_models(quiet=quiet)
    for name, spec in lgb_model_specs.items():
        filepath = os.path.join(FILEPATH, spec["filename"])
        if lgb_model_hashes[name] != hash_of_text_file(filepath):
            return load_lgb_models(quiet=quiet)


reload_lgb_models_if_needed()


def short_pass_success(
    pass_error_diff,
    pos_score_posx,
    pos_score_dnet,
    pos_score_dopp,
    small_cone_angle,
    pass_distance,
    opp_dist_to_line,
    opp_dist_to_active,
):
    return lgb_models["short_pass_success"].predict(
        [
            [
                pass_error_diff,
                pos_score_posx,
                pos_score_dnet,
                pos_score_dopp,
                small_cone_angle,
                pass_distance,
                opp_dist_to_line,
                opp_dist_to_active,
            ]
        ]
    )[0]


def long_pass_success(
    pass_error_diff,
    pos_score_posx,
    pos_score_dnet,
    pos_score_dopp,
    small_cone_angle,
    forward_cone_angle,
    pass_distance,
    opp_dist_to_line,
    opp_dist_to_active,
):
    return lgb_models["long_pass_success"].predict(
        [
            [
                pass_error_diff,
                pos_score_posx,
                pos_score_dnet,
                pos_score_dopp,
                small_cone_angle,
                forward_cone_angle,
                pass_distance,
                opp_dist_to_line,
                opp_dist_to_active,
            ]
        ]
    )[0]


def handle_success(
    pos_score_posx,
    pos_score_dnet,
    pos_score_view,
    pos_score_dopp,
    close_opp_dir_change,
    small_cone_angle,
    angle_diff,
):
    return lgb_models["handle_success"].predict(
        [
            [
                pos_score_posx,
                pos_score_dnet,
                pos_score_view,
                pos_score_dopp,
                close_opp_dir_change,
                small_cone_angle,
                angle_diff,
            ]
        ]
    )[0]


def position_score(
    posx,
    posy,
    dnet,
    view,
    dopp,
):
    # return position_model.predict([[
    #     1.0,  # accidentally had constant in X
    #     posx,
    #     dnet,
    #     view,
    #     dopp,
    # ]])[0]
    return max(
        min(
            307.0343597796267
            + -79.6126832391322 * posx
            + 3.18168640712528 * abs(posy)
            + -170.84343408637883 * dnet
            + 15.225679584757794 * view
            + 87.82098757983785 * dopp
            + -366.6045294401938 * np.log10(posx + 2.0)
            + 80.09708098058961 * posy ** 2
            + -8.123204679740764 * dnet ** 2
            + -6.602709094607595 * view ** 2
            + -528.3954428038661 * dopp ** 2,
            50.0,
        ),
        0,
    )


if __name__ == "__main__":

    # tests:

    short_pass_df = make_short_pass_dataset()

    short_pass_features = lgb_model_specs["short_pass_success"]["features"].keys()
    short_pass_success(**{k: 1.0 for k in short_pass_features})

    long_pass_features = lgb_model_specs["long_pass_success"]["features"].keys()
    long_pass_success(**{k: 1.0 for k in long_pass_features})

    handle_features = lgb_model_specs["handle_success"]["features"].keys()
    handle_success(**{k: 1.0 for k in handle_features})

    print("tests passed")
