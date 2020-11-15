from datetime import time
import numpy as np
from kaggle_environments.envs.football.helpers import Action, GameMode

from . import calculation as calc
from . import config, models
from . import navigation as nav

# this is later set in the agent - I did this for convenience so that
# I wouldn't have to explicitly pass the state around all the time
state = None


class Plan:
    randomisable = False

    def __init__(self, timestep=0):
        self.state = state
        self.timestep = timestep
        self.update_data()
        self.evaluate()

    def __eq__(self, other):
        if isinstance(other, Plan):
            if self.name == other.name:
                self_act = getattr(self, "action_direction", None)
                other_act = getattr(other, "action_direction", None)
                if self_act == other_act:
                    return True
        return False

    def update_data(self):
        self.pos = self.get_position()
        self.dist = nav.dist_1d(self.state.active_pos, self.pos)
        self.position_score()

    def end_of_turn_update(self):
        self.state.write_to_log(self.get_record())

    def position_score(self):

        posx = self.pos[0]
        posy = self.pos[1]
        dnet = calc.get_distance_to_net(self.pos)
        view = calc.get_view_of_net(self.state, self.pos, self.timestep)
        dopp = calc.get_min_opp_distance(self.state, self.pos, self.timestep)
        kopp = calc.get_opp_kernel_density(self.state, self.pos, self.timestep)

        pos_score = models.position_score(
            posx=posx,
            posy=posy,
            dnet=dnet,
            view=view,
            dopp=dopp,
            kopp=kopp,
        )
        # pos_score = pos_score * 0.15 + 25.0 * 0.85

        self.pos_score_data = {
            "posx": posx,
            "posy": posy,
            "dnet": dnet,
            "view": view,
            "dopp": dopp,
            "kopp": kopp,
            "score": pos_score,
        }

        # return 10.0
        return pos_score

    def evaluate(self):
        raise NotImplementedError()

    def get_position(self):
        raise NotImplementedError()

    def get_action(self):
        raise NotImplementedError()

    def get_record(self):
        return {
            "type": "PLAN",
            "plan": self.name,
            "pos": self.pos.round(4),
            "timestep": round(self.timestep, 4),
            "dist": round(self.dist, 4),
            "rand": getattr(self, "rand", 0.0),
            "value": self.value,
            "pos_score_data": self.pos_score_data,
            "eval_data": getattr(self, "eval_data", dict()),
            "action_direction": getattr(self, "action_direction", None),
            "pass_error": getattr(self, "pass_error", -999.0),
            "pass_error_diff": getattr(self, "pass_error_diff", -999.0),
            "shoot_value": getattr(self, "shoot_value", 0.0),
        }


class Move(Plan):
    name = "MOVE"

    def __init__(self, pos, action_direction=None, timestep=0):
        self.pos = pos
        self.action_direction = action_direction
        super().__init__(timestep=timestep)

    def evaluate(self):
        self.value = self.position_score()

    def get_position(self):
        return self.pos

    def get_action(self):

        desired_act = self.action_direction or nav.get_action_direction(
            self.state.active_pos, self.pos
        )
        current_act = nav.get_action_direction(nav.origin, self.state.active_dir)

        # make sure we're heading in the right direction
        if desired_act not in self.state.sticky_actions:
            return desired_act

        # sprint if (and only if) we're going in the right direction
        desired_angle = nav.angle(nav.action_to_vector_map[desired_act])
        current_angle = nav.angle(nav.action_to_vector_map[current_act])
        angle_diff = nav.angle_diff(desired_angle, current_angle)

        if self.state.active_vel > 0.005:  # if we're going fast...
            if angle_diff > 50 and angle_diff < 120:  # ...in a semi-wrong direction
                if self.state.is_sprinting:
                    return Action.ReleaseSprint
            else:
                if not self.state.is_sprinting:
                    return Action.Sprint
        else:
            if not self.state.is_sprinting:
                return Action.Sprint

        return Action.Idle


class MoveWithBall(Plan):
    name = "MOVE_WITH_BALL"
    randomisable = True

    def __init__(self, action_direction, timestep):
        self.action_direction = action_direction
        super().__init__(timestep=timestep)

    def evaluate(self):
        self.value = self.position_score()

        current_dir = self.state.active_dir
        desired_dir = (self.pos - self.state.active_pos) / self.timestep

        # plan is less attractive if we have to change direction and
        # someone is pretty close to us
        close_opp_dir_change = 0.0
        if calc.get_min_opp_distance(self.state, self.state.active_pos) < 0.08:
            close_opp_dir_change = nav.dist_1d(current_dir, desired_dir)

        # plan is less attractive if there's someone in our way
        small_cone_angle = nav.min_opp_angle(
            self.state,
            pos_a=self.state.active_pos,
            pos_b=self.pos,
            ref_dist=0.1,
            ref_offset=-0.02,
        )

        # some inertia:
        angle_diff = nav.angle_diff(nav.angle(current_dir), nav.angle(desired_dir))
        if angle_diff <= 50:
            self.name += "*"

        prb_success = models.handle_success(
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_view=self.pos_score_data["view"],
            pos_score_dopp=self.pos_score_data["dopp"],
            pos_score_kopp=self.pos_score_data["kopp"],
            close_opp_dir_change=close_opp_dir_change,
            small_cone_angle=small_cone_angle,
            angle_diff=angle_diff,
            # timestep=self.timestep,
        )
        self.value *= prb_success ** config.RISK_AVERSION

        # plan is super unattractive if we'd move outside the pitch
        outside_pitch = nav.outside_pitch(self.pos)
        if outside_pitch:
            self.value *= 0.01

        self.eval_data = {
            "timestep": self.timestep,
            "close_opp_dir_change": close_opp_dir_change,
            "small_cone_angle": small_cone_angle,
            "outside_pitch": outside_pitch,
            "angle_diff": angle_diff,
            "prb_success": prb_success,
        }

    def get_position(self):
        desired_dir = nav.action_to_vector_map[self.action_direction]
        return (
            self.state.active_pos
            + self.state.active_dir * 3
            + desired_dir * 0.009 * self.timestep
        )

    def get_action(self):

        # if we don't have the ball yet, just chill out until we do
        if self.state.will_receive_ball_at > 3:
            return Action.ReleaseDirection

        self.state.write_to_log(
            {
                "type": "MOVE_WITH_BALL_ATTEMPT",
                "pos_score_data": self.pos_score_data,
                "eval_data": self.eval_data,
            }
        )

        subplan = Move(self.pos, action_direction=self.action_direction)
        return subplan.get_action()


class ShortPass(Plan):
    name = "SHORT_PASS"
    randomisable = True

    def __init__(
        self,
        player,
        action_direction,
        timestep,
        error=None,
        error_diff=None,
    ):
        self.player = player
        self.action_direction = action_direction
        self.pass_error = error
        self.pass_error_diff = error_diff
        self.follow_through = False
        super().__init__(timestep=timestep)

    def evaluate(self):

        self.value = self.position_score()

        shoot_plan = Shoot(timestep=self.timestep, player=self.player)
        self.shoot_value = shoot_plan.value * 0.2
        self.value += self.shoot_value

        active_pos = self.state.active_pred[self.timestep]
        player_pos = self.state.team_pred[self.player, self.timestep]

        small_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            timestep=self.timestep,
        )
        pass_distance = nav.dist_1d(active_pos, player_pos)
        opp_dist_to_line = calc.opp_density_to_line(
            self.state, active_pos, player_pos, timestep=self.timestep
        )
        opp_dist_to_active = calc.get_min_opp_distance(
            self.state, active_pos, timestep=self.timestep
        )

        prb_success = models.short_pass_success(
            pass_error_diff=self.pass_error_diff,
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_dopp=self.pos_score_data["dopp"],
            pos_score_kopp=self.pos_score_data["kopp"],
            small_cone_angle=small_cone_angle,
            pass_distance=pass_distance,
            opp_dist_to_line=opp_dist_to_line,
            opp_dist_to_active=opp_dist_to_active,
            # timestep=self.timestep,
            # kick_countdown=self.state.player_kicked_countdown_timer,
        )
        self.value *= (prb_success + 0.04) ** config.RISK_AVERSION
        # self.value *= (0.78 + prb_success / 100.0 * 7)

        pos_offside = calc.position_offside(self.state, player_pos, self.timestep)
        if pos_offside:
            self.value *= 0.5

        self.eval_data = {
            "timestep": self.timestep,
            "kick_countdown": self.state.player_kicked_countdown_timer,
            "pass_error": self.pass_error,
            "pass_error_diff": self.pass_error_diff,
            "pos_offside": pos_offside,
            "small_cone_angle": small_cone_angle,
            "pass_distance": pass_distance,
            "opp_dist_to_line": opp_dist_to_line,
            "opp_dist_to_active": opp_dist_to_active,
            "prb_success": prb_success,
        }

    def get_position(self):
        return self.state.team_pred[self.player, self.timestep]

    def get_action(self):

        self.state.put_in_log_queue(
            {
                "type": "SHORT_PASS_ATTEMPT",
                "player": self.player,
                "pos_score_data": self.pos_score_data,
                "eval_data": self.eval_data,
            }
        )

        if self.follow_through:
            if self.action_direction not in self.state.sticky_actions:
                return self.action_direction
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.Idle
        else:
            return Action.ShortPass


class LongPass(Plan):
    name = "LONG_PASS"
    randomisable = True

    def __init__(
        self,
        player,
        action_direction,
        timestep,
        error=None,
        error_diff=None,
    ):
        self.player = player
        self.action_direction = action_direction
        self.pass_error = error
        self.pass_error_diff = error_diff
        self.follow_through = False
        super().__init__(timestep=timestep)

    def evaluate(self):

        self.value = self.position_score()

        shoot_plan = Shoot(timestep=self.timestep, player=self.player)
        self.shoot_value = shoot_plan.value * 0.2
        self.value += self.shoot_value

        active_pos = self.state.active_pred[self.timestep]
        player_pos = self.state.team_pred[self.player, self.timestep]

        small_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=self.state.active_pred[self.timestep],
            pos_b=self.pos,
            timestep=self.timestep,
        )
        forward_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=player_pos,
            pos_b=self.pos,
            timestep=self.timestep,
        )
        pass_distance = nav.dist_1d(active_pos, player_pos)
        opp_dist_to_line = calc.opp_density_to_line(
            self.state, active_pos, self.pos, timestep=self.timestep
        )
        opp_dist_to_active = calc.get_min_opp_distance(
            self.state, active_pos, timestep=self.timestep
        )

        prb_success = models.long_pass_success(
            pass_error_diff=self.pass_error_diff,
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_dopp=self.pos_score_data["dopp"],
            pos_score_kopp=self.pos_score_data["kopp"],
            small_cone_angle=small_cone_angle,
            forward_cone_angle=forward_cone_angle,
            pass_distance=pass_distance,
            opp_dist_to_line=opp_dist_to_line,
            opp_dist_to_active=opp_dist_to_active,
            # timestep=self.timestep,
            # kick_countdown=self.state.player_kicked_countdown_timer,
        )
        self.value *= (prb_success + 0.08) ** config.RISK_AVERSION
        # self.value *= (0.77 + prb_success / 100.0 * 7)

        pos_offside = calc.position_offside(self.state, player_pos, self.timestep)
        if pos_offside:
            self.value *= 0.5

        self.eval_data = {
            "timestep": self.timestep,
            "kick_countdown": self.state.player_kicked_countdown_timer,
            "pass_error": self.pass_error,
            "pass_error_diff": self.pass_error_diff,
            "pos_offside": pos_offside,
            "small_cone_angle": small_cone_angle,
            "forward_cone_angle": forward_cone_angle,
            "pass_distance": pass_distance,
            "opp_dist_to_line": opp_dist_to_line,
            "opp_dist_to_active": opp_dist_to_active,
            "prb_success": prb_success,
        }

    def get_position(self):
        return self.state.team_pred[self.player, self.timestep] + [
            config.LONG_PASS_OFFSET,
            0.0,
        ]

    def get_action(self):

        self.state.put_in_log_queue(
            {
                "type": "LONG_PASS_ATTEMPT",
                "player": self.player,
                "pos_score_data": self.pos_score_data,
                "eval_data": self.eval_data,
            }
        )

        if self.follow_through:
            if self.action_direction not in self.state.sticky_actions:
                return self.action_direction
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.Idle
        else:
            return Action.LongPass


class HighPass(Plan):
    name = "HIGH_PASS"

    def __init__(self):
        self.follow_through = False
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.active_pos[0] > 0.80:
            if abs(self.state.active_pos[1]) > 0.15:
                self.value = 52.0
        if self.state.active_pos[0] < -0.65:
            self.value = 10.0

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        if self.follow_through:
            desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
            if desired_act not in self.state.sticky_actions:
                return desired_act
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.Idle
        else:
            return Action.HighPass


class GoToBall(Plan):
    name = "GO_TO_BALL"

    def __init__(self):
        super().__init__()

    def evaluate(self):
        if self.state.will_receive_ball_at < 8:
            self.value = 0.0
        else:
            self.value = 50.0

    def get_position(self):
        timestep = self.state.can_receive_ball_at
        if np.isnan(timestep):
            timestep = -1
        return self.state.ball_pred[timestep, :2]

    def get_action(self):
        subplan = Move(self.pos)
        return subplan.get_action()


class Shoot(Plan):
    name = "SHOOT"

    def __init__(self, timestep, player=None):
        self.player = player
        self.follow_through = False
        super().__init__(timestep=timestep)

    def evaluate(self):

        if self.player is None:
            self.player = self.state.active_idx

        pos = self.state.team_pred[self.player, self.timestep]
        view_of_net = calc.get_view_of_net(self.state, pos, timestep=self.timestep)
        distance_to_net = calc.get_distance_to_net(pos)

        if config.USE_HARDCODED_SHOT_VALUE:
            prb_success = -1.0
            self.value = (
                10.0 * (pos[0] > 0.0)
                + view_of_net * 100.0 * (distance_to_net < 0.4)
                + 10.0 * (distance_to_net < 0.2)
            )
        else:
            self.value = 400.0 if distance_to_net < 0.35 else 0.0  # TODO: tune this
            prb_success = models.shot_success(
                view_of_net=view_of_net,
                distance_to_net=distance_to_net,
                distance_to_goalie=np.nan,
            )
            self.value *= prb_success ** config.RISK_AVERSION

        self.eval_data = {
            "pos": list(pos.round(4)),
            "view_of_net": view_of_net,
            "distance_to_net": distance_to_net,
            "prb_success": prb_success,
        }
        # temporary diagnostic logging
        self.state.write_to_log(self.eval_data)

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        if self.follow_through:
            desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
            if desired_act not in self.state.sticky_actions:
                return desired_act
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.Idle
        else:
            self.state.put_in_log_queue(
                {
                    "type": "SHOT_ATTEMPT",
                    "eval_data": self.eval_data,
                }
            )
            return Action.Shot


class ChasePlayer(Plan):
    name = "CHASE_PLAYER"

    def __init__(self):
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.right_has_ball:
            self.value = 51.0
        if not self.state.active_has_ball:
            if self.state.opp_will_get_ball_at <= 3:
                self.value = 51.0

    def get_position(self):
        opp = self.state.opp_closest_to_ball
        for t in range(1, self.state.n_step_pred + 1):
            opp_pos = self.state.opp_pred[opp, t]
            offset = nav.normalise_1d(nav.own_goal - opp_pos) * 0.02
            opp_pos += offset
            if nav.dist_1d(opp_pos, self.state.active_pos) < (
                0.014 * t
            ):  # <- too generous?
                return opp_pos
        return self.state.opp_pred[opp, -1]

    def get_action(self):
        subplan = Move(self.pos)
        return subplan.get_action()


class GoalieKick(Plan):
    name = "GOALIE_KICK"

    def evaluate(self):
        self.value = 0.0
        if self.state.goalie_has_ball:
            self.value = 200.0

    def get_position(self):
        return nav.origin

    def get_action(self):
        if Action.TopRight not in self.state.sticky_actions:
            return Action.TopRight
        return Action.HighPass


class CornerKick(Plan):
    name = "CORNER_KICK"

    def evaluate(self):
        self.value = 0.0
        if self.state.game_mode == GameMode.Corner:
            self.value = 200.0

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
        if desired_act not in self.state.sticky_actions:
            return desired_act
        return Action.HighPass


class Breakaway(Plan):
    name = "BREAKAWAY"

    def __init__(self):
        self.follow_through = False
        super().__init__()

    def evaluate(self):

        self.value = 0.0

        if self.state.active_pos[0] > 0.0:
            if self.state.active_pos[0] < 0.8:
                active_dist_to_net = nav.dist_1d(self.state.active_pos, nav.opp_goal)
                opp_dist_to_net = nav.dist_2d(self.state.opp_pos, nav.opp_goal)
                if (active_dist_to_net > opp_dist_to_net).sum() <= 1:
                    self.value = 200.0

    def get_position(self):
        return nav.opp_goal

    def get_action(self):

        if self.follow_through:
            if self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.Idle

        active_pos = self.state.active_pred[6]
        goalie_pos = self.state.opp_pred[0, 6]
        dist_to_goalie = nav.dist_1d(active_pos, goalie_pos)
        if dist_to_goalie <= 0.18:
            desired_act = (
                Action.TopRight if self.state.active_pos[1] > 0 else Action.BottomRight
            )
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                # add some extra metadata from training regular shot action
                pos = self.state.active_pred[self.timestep]
                view_of_net = calc.get_view_of_net(
                    self.state, pos, timestep=self.timestep
                )
                distance_to_net = calc.get_distance_to_net(pos)
                self.state.write_to_log(
                    {
                        "type": "SHOT_ATTEMPT",
                        "eval_data": {
                            "dist_to_goalie": dist_to_goalie,
                            "pos": pos,
                            "view_of_net": view_of_net,
                            "distance_to_net": distance_to_net,
                        },
                    }
                )
                return Action.Shot

        subplan = Move(self.pos)
        return subplan.get_action()


class Kickoff(Plan):
    name = "KICKOFF"

    def __init__(self):
        self.follow_through = False
        super().__init__()

    def evaluate(self):

        self.value = 0.0

        if self.state.ball_pos[0] == 0.0:
            if self.state.ball_pos[1] == 0.0:
                if abs(self.state.active_pos[0]) < 0.03:
                    self.value = 200.0

    def get_position(self):
        return np.array([-0.5, 0.0])

    def get_action(self):
        if self.follow_through:
            if Action.Left not in self.state.sticky_actions:
                return Action.Left
            else:
                return Action.Idle
        else:
            return Action.ShortPass
