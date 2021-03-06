import json
import os
from functools import partial

import lightgbm as lgb
import pandas as pd
from cicada.utils import config, data
from lightgbm.sklearn import LGBMClassifier
from tqdm import tqdm

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

    for col in new_df:
        if new_df[col].isna().all():
            print("dropping column", col, "because all values are nan")
            new_df = new_df.drop(columns=[col])

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
        if "eval_data.dist_to_goalie" in df.columns:
            df = df[df["eval_data.dist_to_goalie"].isna()]

        if "eval_data.pos" in df.columns:
            df["posx"] = df["eval_data.pos"].str[0]
            df["posy"] = df["eval_data.pos"].str[1]

    return df


def short_pass_inner_function(log, n_steps=60):

    filtered_log = data.filter_log(
        log,
        type=(
            "SHORT_PASS_ATTEMPT",
            "NEW_POSSESSION",
            "OPP_POSSESSION",
            "KICK_RELEASE",
            "GOAL_SCORED",
        ),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        if "player" not in df.columns:
            df["player"] = -1

        df["target"] = (
            (df.type == "SHORT_PASS_ATTEMPT")
            & (
                (df.type.shift(-1).isin(["NEW_POSSESSION", "KICK_RELEASE"]))
                | (df.type.shift(-1) == "GOAL_SCORED")
                | ((df.type.shift(-1) == "OPP_POSSESSION") & (df.player.shift(-1) == 0))
            )
            & (df.step.shift(-1) - df.step <= n_steps)
        ).astype(int)
        df = df[df.type == "SHORT_PASS_ATTEMPT"]

    return df


def long_pass_inner_function(log, n_steps=60):

    filtered_log = data.filter_log(
        log,
        type=(
            "LONG_PASS_ATTEMPT",
            "NEW_POSSESSION",
            "OPP_POSSESSION",
            "KICK_RELEASE",
            "GOAL_SCORED",
        ),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        if "player" not in df.columns:
            df["player"] = -1

        df["target"] = (
            (df.type == "LONG_PASS_ATTEMPT")
            & (
                (df.type.shift(-1).isin(["NEW_POSSESSION", "KICK_RELEASE"]))
                | (df.type.shift(-1) == "GOAL_SCORED")
                | ((df.type.shift(-1) == "OPP_POSSESSION") & (df.player.shift(-1) == 0))
            )
            & (df.step.shift(-1) - df.step <= n_steps)
        ).astype(int)
        df = df[df.type == "LONG_PASS_ATTEMPT"]

    return df


def handle_inner_function(log, n_steps=10):

    filtered_log = data.filter_log(
        log,
        type=(
            "MOVE_WITH_BALL_ATTEMPT",
            "LOST_POSSESSION",
            "NEW_POSSESSION",
            "OPP_POSSESSION",
            "SHOT_ATTEMPT",
        ),
    )
    df = data.parse_log_to_df(filtered_log)

    if len(df):

        df["target"] = 1

        bad_steps = df.step[df.type.isin(("LOST_POSSESSION", "OPP_POSSESSION"))]
        for step in bad_steps:

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
            df.loc[mask, "reward"] = -0.025

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
        "validation_size": 100_000,
        "features": {
            "pass_error_diff": "eval_data.pass_error_diff",
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_view": "pos_score_data.view",
            "pos_score_dopp": "pos_score_data.dopp",
            "pos_score_kopp": "pos_score_data.kopp",
            "small_cone_angle": "eval_data.small_cone_angle",
            "tiny_cone_angle": "eval_data.tiny_cone_angle",
            "micro_cone_angle": "eval_data.micro_cone_angle",
            "reverse_tiny_cone_angle": "eval_data.reverse_tiny_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
            "opp_dist_to_active_now": "eval_data.opp_dist_to_active_now",
            "opp_dist_now": "eval_data.opp_dist_now",
            "angle_diff": "eval_data.angle_diff",
            "angle_to_sticky": "eval_data.angle_to_sticky",
            "active_vel": "eval_data.active_vel",
            "player_vel": "eval_data.player_vel",
            "one_time_kick": "one_time_kick",
        },
        "monotone_constraints": {
            "pos_score_dopp": 1,
            "pos_score_kopp": -1,
            "small_cone_angle": 1,
            "opp_dist_to_line": -1,
            "opp_dist_to_active": 1,
            "opp_dist_now": 1,
            "angle_diff": -1,
        },
        "default_prediction": 0.8,
    },
    "long_pass_success": {
        "filename": "long_pass_model.txt",
        "dataset": make_long_pass_dataset,
        "validation_size": 100_000,
        "features": {
            "pass_error_diff": "eval_data.pass_error_diff",
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_view": "pos_score_data.view",
            "pos_score_dopp": "pos_score_data.dopp",
            "pos_score_kopp": "pos_score_data.kopp",
            "small_cone_angle": "eval_data.small_cone_angle",
            "forward_cone_angle": "eval_data.forward_cone_angle",
            "tiny_cone_angle": "eval_data.tiny_cone_angle",
            "pass_distance": "eval_data.pass_distance",
            "opp_dist_to_line": "eval_data.opp_dist_to_line",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
            "opp_dist_to_active_now": "eval_data.opp_dist_to_active_now",
            "opp_dist_now": "eval_data.opp_dist_now",
            "angle_diff": "eval_data.angle_diff",
            "angle_to_sticky": "eval_data.angle_to_sticky",
            "active_vel": "eval_data.active_vel",
            "player_vel": "eval_data.player_vel",
            "one_time_kick": "one_time_kick",
        },
        "monotone_constraints": {
            "pos_score_dopp": 1,
            "pos_score_kopp": -1,
            "small_cone_angle": 1,
            "forward_cone_angle": 1,
            "opp_dist_to_line": -1,
            "opp_dist_to_active": 1,
            "angle_diff": -1,
        },
        "default_prediction": 0.8,
    },
    "handle_success": {
        "filename": "handle_model.txt",
        "dataset": make_handle_dataset,
        "validation_size": 1_000_000,
        "features": {
            "pos_score_posx": "pos_score_data.posx",
            "pos_score_dnet": "pos_score_data.dnet",
            "pos_score_view": "pos_score_data.view",
            "close_opp_dir_change": "eval_data.close_opp_dir_change",
            "small_cone_angle": "eval_data.small_cone_angle",
            "micro_cone_angle": "eval_data.micro_cone_angle",
            "opp_dist_to_active": "eval_data.opp_dist_to_active",
            "opp_dist_to_active_now": "eval_data.opp_dist_to_active_now",
            "active_vel": "eval_data.active_vel",
            "relative_distance": "eval_data.relative_distance",
            "angle_diff": "eval_data.angle_diff",
            "angle_to_sticky": "eval_data.angle_to_sticky",
        },
        "monotone_constraints": {
            "pos_score_view": 1,
            "pos_score_dopp": 1,
            "pos_score_kopp": -1,
            "close_opp_dir_change": -1,
            "small_cone_angle": 1,
            "opp_dist_to_active": 1,
            "opp_dist_to_active_now": 1,
            "angle_diff": -1,
        },
        "default_prediction": 0.8,
    },
    "shot_success": {
        "filename": "shot_model.txt",
        "dataset": make_shot_dataset,
        "validation_size": 10_000,
        "features": {
            "posx": "posx",
            "posy": "posy",
            "view_of_net": "eval_data.view_of_net",
            "distance_to_net": "eval_data.distance_to_net",
            "shooter_kopp": "eval_data.shooter_kopp",
        },
        "monotone_constraints": {"view_of_net": 1},
        "default_prediction": 0.1,
    },
}


def fit_lgb_model(model_spec, early_stopping_rounds=10):
    ms = model_spec
    print("loading data using", ms["filename"])
    df = ms["dataset"]()
    print("converting all feature columns to float32")
    for col in ms["features"].values():
        df[col] = df[col].astype("float32")
    print(df.describe().T)
    n_targets_with_null = df["target"].isna().sum()
    print("dropping", n_targets_with_null, "rows with null in target")
    df = df[df["target"].notna()]
    monotone_constraints = [
        ms["monotone_constraints"].get(col, 0) for col in ms["features"].keys()
    ]
    print("monotone constraints:", monotone_constraints)
    model = LGBMClassifier(
        n_estimators=5_000,
        num_leaves=11,
        learning_rate=0.01,
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="advanced",
    )
    X = df[ms["features"].values()]
    y = df["target"]
    n_games = df["game_id"].max() + 1
    game_pct = (df["game_id"] + 1) / n_games
    w = config.GAME_WEIGHTING_FACTOR + (1 - config.GAME_WEIGHTING_FACTOR) * game_pct
    eval_size = ms["validation_size"]
    X_tr, X_te = X.iloc[:-eval_size], X.iloc[-eval_size:]
    y_tr, y_te = y.iloc[:-eval_size], y.iloc[-eval_size:]
    w_tr, w_te = w.iloc[:-eval_size], w.iloc[-eval_size:]
    eval_set = [(X_te.values, y_te.values)]
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=eval_set,
        eval_sample_weight=[w_te],
        early_stopping_rounds=early_stopping_rounds,
        verbose=early_stopping_rounds,
    )
    print("refitting model with full dataset")
    model.set_params(n_estimators=model.best_iteration_)
    model.fit(X, y, sample_weight=w)
    pred = pd.Series(model.predict_proba(X)[:, 1])
    print("distribution of predictions:")
    print(pred.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
    feature_importances = [
        (feature, importance)
        for feature, importance in zip(
            model.booster_.feature_name(),
            model.booster_.feature_importance(importance_type="gain"),
        )
    ]
    print("feature importance (by gain):")
    for feature, importance in sorted(feature_importances, key=lambda row: -row[1]):
        print(f"    {feature}: {importance}")
    filepath = os.path.join(FILEPATH, "models", ms["filename"])
    print("saving to", filepath)
    model.booster_.save_model(filepath)


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
    func_name = f"{name}_pred_func"
    function_string = f"def {func_name}({feature_names}): return lgb_models['{name}'].predict([[{feature_names}]])[0]"  # noqa
    exec(function_string)
    return locals()[func_name]


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
            41.88600209739781
            + -26.87241792314 * dnet
            + -70.30815593524 * abs(posy)
            + 50.377999345915 * dnet * abs(posy)
            + 48.783552591921 * view
            + -61.53807858830 * view ** 2
            + 13.110810164865 * view ** 3,
            50.0,
        ),
        0,
    )


if __name__ == "__main__":

    # test prediction functions
    for name, spec in lgb_model_specs.items():
        print(name, "-", end=" ")
        features = spec["features"].keys()
        func = make_predict_function(name)
        pred = func(**{k: 1.0 for k in features})
        print(pred)

    print("tests passed ✅")

    print("training models on existing data")
    for name, spec in lgb_model_specs.items():
        fit_lgb_model(spec)
