from kaggle_environments.envs.football.helpers import (
    Action,
    GameMode,
    PlayerRole,
)

from .utils.data import State, clean_observation

from copy import deepcopy


class SimpleAgent:
    def __init__(self):
        self.events = list()
        self.step = -1

    def action_wrapper(self, obs):

        obs = deepcopy(obs)
        obs = clean_observation(obs)

        action = self.action(obs)

        self.events.append(
            {
                "step": self.step,
                "type": "ACTION",
                "action": action,
            }
        )

        return [action.value]

    def action(self, obs):

        self.step += 1

        s = State(obs)

        self.events.append(
            {
                "step": self.step,
                "type": "BALL",
                "ball_pos_x": s.ball_pos[0],
                "ball_pos_y": s.ball_pos[1],
                "ball_pos_z": s.ball_pos[2],
                "ball_dir_x": s.ball_dir[0],
                "ball_dir_y": s.ball_dir[1],
                "ball_dir_z": s.ball_dir[2],
                "ball_rot_x": s.ball_rot[0],
                "ball_rot_y": s.ball_rot[1],
                "ball_rot_z": s.ball_rot[2],
            }
        )

        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        controlled_player_pos = obs["left_team"][obs["active"]]
        if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_team"] == 0:
            if Action.Right not in obs["sticky_actions"]:
                return Action.Right
            if controlled_player_pos[0] > 0.5:
                return Action.Shot
            if controlled_player_pos[0] < -0.1:
                if Action.Right in obs["sticky_actions"]:
                    return Action.HighPass
            return Action.Idle
        else:
            dist_horizontal = obs["ball"][0] - controlled_player_pos[0]
            dist_vertical = obs["ball"][1] - controlled_player_pos[1]
            if min(abs(dist_horizontal), abs(dist_vertical)) > 0.01:
                if dist_horizontal > 0:
                    if dist_vertical > 0:
                        return Action.BottomRight
                    else:
                        return Action.TopRight
                else:
                    if dist_vertical > 0:
                        return Action.BottomLeft
                    else:
                        return Action.TopLeft
            if abs(dist_horizontal) > abs(dist_vertical):
                if dist_horizontal > 0:
                    if Action.Right not in obs["sticky_actions"]:
                        return Action.Right
                else:
                    if Action.Left not in obs["sticky_actions"]:
                        return Action.Left
            else:
                if dist_vertical < 0:
                    if Action.Top not in obs["sticky_actions"]:
                        return Action.Top
                else:
                    if Action.Bottom not in obs["sticky_actions"]:
                        return Action.Bottom
            return Action.Idle
