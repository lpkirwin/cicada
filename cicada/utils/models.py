import json
import os

# import joblib
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

from . import data
from cicada import training

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
            existing_length = len(existing_df)
        except FileNotFoundError:
            existing_length = 0
    else:
        existing_length = 0

    dfs = list()
    game_id = existing_length

    with training.get_log_file() as file:
        for i, log in tqdm(enumerate(file), total=training.get_n_games()):

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
        "features": [
            "pass_error_diff",
            "pos_score_posx",
            "pos_score_dnet",
            "pos_score_dopp",
            "small_cone_angle",
            "pass_distance",
            "opp_dist_to_line",
            "opp_dist_to_active",
        ],
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
        "features": [
            "pass_error_diff",
            "pos_score_posx",
            "pos_score_dnet",
            "pos_score_dopp",
            "small_cone_angle",
            "forward_cone_angle",
            "pass_distance",
            "opp_dist_to_line",
            "opp_dist_to_active",
        ],
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
        "features": [
            "pos_score_posx",
            "pos_score_dnet",
            "pos_score_view",
            "pos_score_dopp",
            "close_opp_dir_change",
            "small_cone_angle",
            "angle_diff",
        ],
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
    grid_search = GridSearchCV(
        estimator=ms["class"](),
        param_grid=ms["grid"],
        scoring=ms["scoring"],
        verbose=1,
        n_jobs=5,
    )
    X = df[ms["features"]]
    y = df["target"]
    print("fitting", ms["filename"])
    grid_search.fit(X, y)
    print("best params:", grid_search.best_params_)
    pred = grid_search.predict_proba(X)[:, 1]
    print("distribution of predictions:", pd.Series(pred).describe())
    filepath = os.path.join(FILEPATH, ms["filename"])
    print("saving to", filepath)
    grid_search.best_estimator_.booster_.save_model(filepath)


class PlaceholderModel:
    def predict(*args, **kwargs):
        return [0.5]


def load_lgb_models():

    global lgb_models
    lgb_models = dict()

    for name, spec in lgb_model_specs.items():

        try:
            model = lgb.Booster(model_file=os.path.join(FILEPATH, spec["filename"]))
            print("model loaded from", spec["filename"])
        except lgb.basic.LightGBMError:
            print("failed to load model from", spec["filename"], "using placeholder")
            model = PlaceholderModel()

        lgb_models[name] = model


load_lgb_models()


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
            181.1331806395123
            + -44.42213922157399 * posx
            + 0.0 * posy
            + -111.73963266453272 * dnet
            + 13.583451757803484 * view
            + 82.9471650898806 * dopp
            + -174.70638402206689 * np.log10(posx + 2.0)
            + 0.0 * posy ** 2
            + 3.0419922087739772 * dnet ** 2
            + -5.5667928903785215 * view ** 2
            + -505.0449817847152 * dopp ** 2,
            50.0,
        ),
        0,
    )
