import os
import lightgbm as lgb
import pandas as pd
# import numpy as np

FILEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_PASS_MODEL_PATH = os.path.join(FILEPATH, "short_pass_model.txt")
LONG_PASS_MODEL_PATH = os.path.join(FILEPATH, "long_pass_model.txt")
HANDLE_MODEL_PATH = os.path.join(FILEPATH, "handle_model.txt")

short_pass_model = lgb.Booster(model_file=SHORT_PASS_MODEL_PATH)
long_pass_model = lgb.Booster(model_file=LONG_PASS_MODEL_PATH)
handle_model = lgb.Booster(model_file=HANDLE_MODEL_PATH)


def short_pass_success(
    pass_error_diff,
    pos_score_posx,
    pos_score_dnet,
    pos_score_dopp,
    small_cone_angle,
    pass_distance,
):
    return short_pass_model.predict([[
        pass_error_diff,
        pos_score_posx,
        pos_score_dnet,
        pos_score_dopp,
        small_cone_angle,
        pass_distance,
    ]])[0]
    # return max(
    #     min(
    #         1.9341
    #         + 0.0018 * pass_error_diff
    #         + -1.3993 * pos_score_posx
    #         + -1.2315 * pos_score_dnet
    #         + -0.0061 * pos_score_dopp
    #         + 0.0011 * small_cone_angle,
    #         1,
    #     ),
    #     0,
    # )


def long_pass_success(
    pass_error_diff,
    pos_score_posx,
    pos_score_dnet,
    pos_score_dopp,
    small_cone_angle,
    forward_cone_angle,
    pass_distance,
):
    return long_pass_model.predict([[
        pass_error_diff,
        pos_score_posx,
        pos_score_dnet,
        pos_score_dopp,
        small_cone_angle,
        forward_cone_angle,
        pass_distance,
    ]])[0]
    # return max(
    #     min(
    #         1.3669
    #         + 0.0016 * pass_error_diff
    #         + -0.7702 * pos_score_posx
    #         + -0.7876 * pos_score_dnet
    #         + -0.2424 * pos_score_dopp
    #         + 0.0015 * small_cone_angle
    #         + 2.169e-05 * forward_cone_angle,
    #         1,
    #     ),
    #     0,
    # )


def handle_failure(
    pos_score_posx,
    pos_score_dnet,
    pos_score_view,
    pos_score_dopp,
    close_opp_dir_change,
    small_cone_angle,
    angle_diff,
):
    return handle_model.predict([[
        pos_score_posx,
        pos_score_dnet,
        pos_score_view,
        pos_score_dopp,
        close_opp_dir_change,
        small_cone_angle,
        angle_diff,
    ]])[0]
    # return max(
    #     min(
    #         0.1428
    #         + -0.0604 * pos_score_posx
    #         + -0.0182 * pos_score_dnet
    #         + 0.0983 * pos_score_view
    #         + -0.3847 * pos_score_dopp
    #         + 1.1022 * close_opp_dir_change
    #         + -0.0005 * small_cone_angle
    #         + 0.0003 * angle_diff,
    #         1,
    #     ),
    #     0,
    # )
