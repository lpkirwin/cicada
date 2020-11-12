import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from kaggle_environments.envs.football.helpers import (
    Action,
    GameMode,
    PlayerRole,
    sticky_index_to_action,
)

# from . import plans
from . import calculation as calc
from . import config
from . import navigation as nav

FILEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(FILEPATH, "log.jsonl")
SCORE_PATH = os.path.join(FILEPATH, "score.csv")


def clean_observation(obs):
    # same as 'human_readable_agent' from kaggle-environments for now
    obs = obs["players_raw"][0]
    obs["sticky_actions"] = [  # set or list?
        sticky_index_to_action[nr]
        for nr, action in enumerate(obs["sticky_actions"])
        if action
    ]
    obs["game_mode"] = GameMode(obs["game_mode"])
    if "designated" in obs:
        del obs["designated"]
    obs["left_team_roles"] = [PlayerRole(role) for role in obs["left_team_roles"]]
    obs["right_team_roles"] = [PlayerRole(role) for role in obs["right_team_roles"]]
    return obs


def step_to_obs(step):
    obs = step[0]["observation"]
    obs = clean_observation(obs)
    obs["last_action"] = [Action(act) for act in step[0]["action"]]
    return obs


class State:
    def __init__(self, obs):
        self.n_step_pred = 25
        self.log = [[]]
        self.log_queue = list()
        self.score = [0, 0]
        self.ball_owned_player = None
        self.left_has_ball = False
        self.active_idx = None
        self.player_kicked = -1
        self.player_kicked_countdown_timer = 0
        self.follow_through_pos = nav.invalid
        self.follow_through_plan = None
        self.step = -1
        self.start_of_turn_update(obs)

    def put_in_log_queue(self, record):
        record["queued_at"] = self.step
        self.log_queue.append(record)

    def write_log_queue_to_log(self, max_age=10, min_age=1, filters=None):
        filters = filters or dict()
        sorted_queue = sorted(self.log_queue, key=lambda r: r["queued_at"])
        rec_to_write = None
        for rec in sorted_queue:
            age = self.step - rec["queued_at"]
            if age > max_age or age < min_age:
                continue
            keep = False
            for key, value in filters.items():
                if rec.get(key) == value:
                    keep = True
            if rec_to_write is None:
                keep = True
            if keep:
                rec_to_write = rec
        if rec_to_write is not None:
            self.write_to_log(rec_to_write)
        else:
            self.write_to_log({"type": "KICK_WITH_NO_ATTEMPT_EVENT"})
        self.log_queue = list()

    def write_to_log(self, record):
        record["step"] = self.step
        self.log[-1].append(record)

    def start_of_turn_update(self, obs):

        self.step += 1

        self.obs = obs

        self.game_mode = obs["game_mode"]

        if obs["score"][0] > self.score[0]:
            self.write_to_log({"type": "GOAL_SCORED"})
        self.score = obs["score"]

        if obs["ball_owned_team"] == 0:
            if obs["ball_owned_player"] != self.ball_owned_player:
                self.write_to_log(
                    {"type": "NEW_POSSESSION", "player": obs["ball_owned_player"]}
                )
        self.ball_owned_player = obs["ball_owned_player"]

        # if we don't have possession, but we had it last turn:
        if obs["ball_owned_team"] != 0 and self.left_has_ball:
            if not self.player_kicked_countdown_timer > 0:
                self.write_to_log({"type": "LOST_POSSESSION"})

        if obs["active"] != self.active_idx:
            if self.player_kicked_countdown_timer > 0:
                self.write_to_log({"type": "KICK_RELEASE"})
                self.write_log_queue_to_log(filters={"player": obs["active"]})
                # ^ may have evaluation data for passes to multiple players,
                # but we only want to keep records for the player we actually
                # end up attempting the pass to (it's possible that we pass
                # to a player that we don't have an 'attempt' event to, in
                # which case we won't record the pass at all in our training
                # data)

        self.ball_pos = np.array(obs["ball"])
        self.ball_dir = np.array(obs["ball_direction"])
        self.ball_rot = np.array(obs["ball_rotation"])

        self.team_n_players = len(obs["left_team"])
        self.opp_n_players = len(obs["right_team"])

        self.team_pos = np.array(obs["left_team"])
        self.team_dir = np.array(obs["left_team_direction"])
        self.team_vel = np.linalg.norm(self.team_dir, axis=1)

        self.opp_pos = np.array(obs["right_team"])
        self.opp_dir = np.array(obs["right_team_direction"])
        self.opp_vel = np.linalg.norm(self.opp_dir, axis=1)

        self.active_idx = obs["active"]
        self.active_pos = self.team_pos[self.active_idx]
        self.active_dir = self.team_dir[self.active_idx]
        self.active_deg = nav.angle(self.active_dir)
        self.active_vel = np.linalg.norm(self.active_dir)
        self.active_has_ball = obs["ball_owned_player"] == obs["active"]

        self.left_has_ball = obs["ball_owned_team"] == 0
        self.right_has_ball = obs["ball_owned_team"] == 1

        self.goalie_has_ball = self.left_has_ball and obs["ball_owned_player"] == 0

        self.sticky_actions = obs["sticky_actions"]
        self.is_sprinting = Action.Sprint in self.sticky_actions

        self.opp_dist_to_active = nav.dist_2d(self.active_pos, self.opp_pos)

        self.opp_closest_to_ball = np.argmin(
            nav.dist_2d(self.ball_pos[:2], self.opp_pos)
        )

        # # offside rule
        # opp_second_last_x = sorted(self.opp_pos[:, 0])[-2]
        # self.team_offside = (
        #     self.team_pos[:, 0] > opp_second_last_x
        # ) & (self.ball_pos[0] < opp_second_last_x)

        # distance between all pairs of my players
        x_diff = np.subtract.outer(self.team_pos[:, 0], self.team_pos[:, 0])
        y_diff = np.subtract.outer(self.team_pos[:, 1], self.team_pos[:, 1])
        self.player_distance_matrix = (x_diff ** 2 + y_diff ** 2) ** 0.5

        # distance between my players and opp players
        x_diff = np.subtract.outer(self.team_pos[:, 0], self.opp_pos[:, 0])
        y_diff = np.subtract.outer(self.team_pos[:, 1], self.opp_pos[:, 1])
        self.opp_distance_matrix = (x_diff ** 2 + y_diff ** 2) ** 0.5

        # distance between long pass targets and opp players
        x_diff = np.subtract.outer(
            self.team_pos[:, 0] + config.LONG_PASS_OFFSET,
            self.opp_pos[:, 0],
        )
        y_diff = np.subtract.outer(self.team_pos[:, 1], self.opp_pos[:, 1])
        self.opp_distance_matrix_long = (x_diff ** 2 + y_diff ** 2) ** 0.5

        # minimum opp distance for each player
        self.min_opp_distance = self.opp_distance_matrix.min(axis=1)
        self.min_opp_distance_long = self.opp_distance_matrix_long.min(axis=1)

        # ball predictions
        self.ball_pred = np.zeros(shape=(self.n_step_pred + 1, 3))
        pos_ = self.ball_pos.copy()
        dir_ = self.ball_dir.copy()
        for t in range(self.n_step_pred + 1):
            self.ball_pred[t] = pos_
            pos_ += dir_  # update ball based on current direction
            pos_[2] += -0.05  # little less z because gravity
            dir_[0] += dir_[0] * -0.05  # friction
            dir_[1] += dir_[1] * -0.05  # friction
            dir_[2] += dir_[2] * -0.01 - 0.08  # gravity again

        # my player predictions
        self.team_pred = np.zeros(shape=(self.team_n_players, self.n_step_pred + 1, 2))
        for p in range(self.team_n_players):
            pos_ = self.team_pos[p].copy()
            dir_ = self.team_dir[p].copy()
            for t in range(self.n_step_pred + 1):
                self.team_pred[p, t] = pos_
                pos_ += dir_  # update player based on current direction

        # opponent player predictions
        self.opp_pred = np.zeros(shape=(self.opp_n_players, self.n_step_pred + 1, 2))
        for o in range(self.opp_n_players):
            pos_ = self.opp_pos[o].copy()
            dir_ = self.opp_dir[o].copy()
            for t in range(self.n_step_pred + 1):
                self.opp_pred[o, t] = pos_
                pos_ += dir_  # update player based on current direction

        # will ball collide with opponent within n steps
        self.opp_will_get_ball = np.array([False] * (self.n_step_pred + 1))
        if self.right_has_ball:
            self.opp_will_get_ball[0] = True
        for t in range(1, self.n_step_pred + 1):
            for o in range(self.opp_n_players):
                opp_pos = self.opp_pred[o, t]
                ball_pos = self.ball_pred[t, :2]
                if nav.dist_1d(opp_pos, ball_pos) < 0.02:
                    self.opp_will_get_ball[t] = True
        try:
            self.opp_will_get_ball_at = min(np.where(self.opp_will_get_ball)[0])
        except ValueError:
            self.opp_will_get_ball_at = np.nan

        # will active player collide with opponent within n steps
        self.will_collide_with_opp = np.array([False] * (self.n_step_pred + 1))
        for t in range(1, self.n_step_pred + 1):
            for o in range(self.opp_n_players):
                opp_pos = self.opp_pred[o, t]
                active_pos = self.team_pred[self.active_idx, t]
                if nav.dist_1d(opp_pos, active_pos) < 0.02:
                    self.will_collide_with_opp[t] = True
        try:
            self.will_collide_with_opp_at = min(np.where(self.will_collide_with_opp)[0])
        except ValueError:
            self.will_collide_with_opp_at = np.nan

        # will active player receive ball within n steps with current direction
        self.will_receive_ball = np.array([False] * (self.n_step_pred + 1))
        if self.active_has_ball:
            self.will_receive_ball[0] = True
        for t in range(1, self.n_step_pred + 1):
            ball_pos = self.ball_pred[t, :2]
            active_pos = self.team_pred[self.active_idx, t]
            if nav.dist_1d(ball_pos, active_pos) < 0.02:
                if self.ball_pred[t, 2] < 1.2:
                    self.will_receive_ball[t] = True
        try:
            self.will_receive_ball_at = min(np.where(self.will_receive_ball)[0])
        except ValueError:
            self.will_receive_ball_at = np.nan

        # can active player receive ball within n steps moving at x per step
        self.can_receive_ball = np.array([False] * (self.n_step_pred + 1))
        for t in range(1, self.n_step_pred + 1):
            ball_pos = self.ball_pred[t, :2]
            if nav.dist_1d(ball_pos, self.active_pos) < (0.014 * t):  # too generous?
                if self.ball_pred[t, 2] < 1.5:  # how tall are players?
                    self.can_receive_ball[t] = True
        try:
            self.can_receive_ball_at = min(np.where(self.can_receive_ball)[0])
        except ValueError:
            self.can_receive_ball_at = np.nan

        # view of net
        self.active_intercepts = list()
        self.active_view_of_net = calc.get_view_of_net(
            self, self.active_pos, timestep=0, log_intercepts=True
        )

        # error logging for pass targeting
        self.sticky_action_pass_error = np.ones_like(self.team_pos) * 999.0
        for action in self.sticky_actions:
            if action in nav.action_to_vector_map:
                self.sticky_action_pass_error = nav.pass_error(
                    self.team_pos, self.active_idx, action
                )

    def end_of_turn_update(self):
        self.player_kicked_countdown_timer = max(
            self.player_kicked_countdown_timer - 1, 0
        )
        self.write_to_log(
            {
                "type": "END_OF_TURN",
                "game_mode": self.game_mode,
                "elapsed_time": self.elapsed_time,
                "action": self.action,
                "kick_countdown": self.player_kicked_countdown_timer,
                "kick_player": (
                    self.player_kicked
                    if self.player_kicked_countdown_timer > 0
                    else "n/a"
                ),
                "ball_pred": self.ball_pred,
                "active_pred": self.team_pred[self.active_idx],
                "will_receive_ball_at": self.will_receive_ball_at,
                "can_receive_ball_at": self.can_receive_ball_at,
                "will_collide_with_opp_at": self.will_collide_with_opp_at,
                "active_view_of_net": self.active_view_of_net,
                "active_intercepts": self.active_intercepts,
                "min_opp_distance": self.min_opp_distance,
                "sticky_action_pass_error": self.sticky_action_pass_error,
            }
        )
        self.log.append(list())


def filter_log_step(log_step, **kwargs):
    """Returns filtered log step (creates a copy)"""
    out = list()
    for rec in log_step:
        for k, values in kwargs.items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            for v in values:
                if rec[k] == v:
                    out.append(deepcopy(rec))
    return out


def filter_log(log, **kwargs):
    """Filters each step in a log (creates a copy, drops empty steps)"""
    # import pdb; pdb.set_trace()
    out = list()
    for log_step in log:
        filtered_step = filter_log_step(log_step, **kwargs)
        if len(filtered_step):
            out.append(filtered_step)
    return out


def parse_log_to_df(log, **kwargs):
    rows = list()
    for step in log:
        rows.extend(step)
    return pd.json_normalize(rows)


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
