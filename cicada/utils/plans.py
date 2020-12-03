import numpy as np
from kaggle_environments.envs.football.helpers import Action, GameMode

from . import calculation as calc
from . import config, models
from . import navigation as nav

# this value is later overwritten in the agent module - I did this for
# convenience so that I wouldn't have to explicitly pass the state
# around all the time
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
            if self.name[:5] == other.name[:5]:
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

        self.pos_score_data = {
            "posx": posx,
            "posy": posy,
            "dnet": dnet,
            "view": view,
            "dopp": dopp,
            "kopp": kopp,
            "score": pos_score,
        }

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
            "kick_hold_length": getattr(self, "kick_hold_length", 0),
            "player": getattr(self, "player", ""),
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

        desired_angle = nav.angle(nav.action_to_vector_map[desired_act])
        current_angle = nav.angle(nav.action_to_vector_map[current_act])
        angle_diff = nav.angle_diff(desired_angle, current_angle)

        # if we're basically already where we want to be and heading
        # in roughly the right direction
        next_dist_to_target = nav.dist_1d(self.pos, self.state.active_pred[1])
        if angle_diff < 30:
            if next_dist_to_target < 0.02:
                if self.state.is_sprinting:
                    return Action.ReleaseSprint
                if next_dist_to_target < 0.01:
                    return Action.ReleaseDirection

        # sprint if (and only if) we're going in the right direction
        if self.state.active_vel > 0.005:  # if we're going fast...
            if angle_diff > 50 and angle_diff < 120:  # ...in a semi-wrong direction
                if self.state.is_sprinting:
                    return Action.ReleaseSprint
            else:
                if not self.state.is_sprinting:
                    return Action.Sprint
        else:  # if we're going slow
            if not self.state.is_sprinting:
                return Action.Sprint

        return Action.ReleaseDribble


class MoveWithBall(Plan):
    name = "MOVE_WITH_BALL"
    randomisable = True

    def __init__(self, action_direction, timestep):
        self.action_direction = action_direction
        super().__init__(timestep=timestep)

    def evaluate(self):
        self.value = self.position_score()

        current_dir = self.state.active_dir
        sticky_dir = nav.action_to_vector_map[self.action_direction]
        desired_dir = (self.pos - self.state.active_pos) / self.timestep

        active_pos = self.state.active_pred[self.timestep]

        # plan is less attractive if we have to change direction and
        # someone is pretty close to us
        close_opp_dir_change = 0.0
        if calc.get_min_opp_distance(self.state, self.state.active_pos) < 0.08:
            close_opp_dir_change = nav.dist_1d(current_dir, desired_dir)
        small_cone_angle = nav.min_opp_angle(
            self.state,
            pos_a=self.state.active_pos,
            pos_b=self.pos,
            ref_dist=0.1,
            ref_offset=-0.02,
        )
        micro_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            ref_dist=0.05,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        opp_dist_to_active = calc.get_min_opp_distance(
            self.state, active_pos, timestep=self.timestep
        )
        opp_dist_to_active_now = calc.get_min_opp_distance(
            self.state, active_pos, timestep=0
        )
        angle_diff = nav.angle_diff(nav.angle(current_dir), nav.angle(desired_dir))
        angle_to_sticky = nav.angle_diff(nav.angle(sticky_dir), nav.angle(desired_dir))
        relative_distance = calc.get_min_opp_distance(
            self.state, self.pos, timestep=0
        ) - nav.dist_1d(self.state.active_pos, self.pos)

        prb_success = models.handle_success(
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_view=self.pos_score_data["view"],
            close_opp_dir_change=close_opp_dir_change,
            small_cone_angle=small_cone_angle,
            micro_cone_angle=micro_cone_angle,
            opp_dist_to_active=opp_dist_to_active,
            opp_dist_to_active_now=opp_dist_to_active_now,
            active_vel=self.state.active_vel,
            relative_distance=relative_distance,
            angle_diff=angle_diff,
            angle_to_sticky=angle_to_sticky,
        )
        self.value *= prb_success

        # plan is super unattractive if we'd move outside the pitch
        outside_pitch = nav.outside_pitch(self.pos)
        if outside_pitch:
            self.value *= 0.01

        self.eval_data = {
            "timestep": self.timestep,
            "close_opp_dir_change": close_opp_dir_change,
            "small_cone_angle": small_cone_angle,
            "micro_cone_angle": micro_cone_angle,
            "opp_dist_to_active": opp_dist_to_active,
            "opp_dist_to_active_now": opp_dist_to_active_now,
            "angle_diff": angle_diff,
            "angle_to_sticky": angle_to_sticky,
            "active_vel": self.state.active_vel,
            "relative_distance": relative_distance,
            "outside_pitch": outside_pitch,
            "prb_success": prb_success,
        }

    def get_position(self):
        desired_dir = nav.action_to_vector_map[self.action_direction]
        steps = max(self.timestep - 1 * self.state.will_receive_ball_at, 0)
        return (
            self.state.active_pos
            + self.state.active_dir * 4
            + desired_dir * (0.0065 * steps + 0.001)
        )

    def get_action(self):

        # if we don't have the ball yet, just chill out until we do
        if self.state.will_receive_ball_at > 3:
            if self.state.is_sprinting:
                return Action.ReleaseSprint
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
        kick_hold_length,
        error=None,
        error_diff=None,
    ):
        self.player = player
        self.action_direction = action_direction
        self.pass_error = error
        self.pass_error_diff = error_diff
        self.kick_hold_length = kick_hold_length
        self.follow_through = False
        super().__init__(timestep=timestep)

    def evaluate(self):

        self.value = self.position_score()
        self.value += config.SHORT_PASS_BONUS

        shoot_plan = Shoot(timestep=self.timestep, player=self.player)
        self.shoot_value = shoot_plan.value * 0.15
        self.value += self.shoot_value

        active_pos = self.state.active_pred[self.timestep]
        player_pos = self.state.team_pred[self.player, self.timestep]

        current_dir = self.state.active_dir
        sticky_dir = nav.action_to_vector_map[self.action_direction]
        desired_dir = (self.pos - self.state.active_pos) / (self.timestep + 1)

        small_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            timestep=self.timestep,
        )
        tiny_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            ref_dist=0.1,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        micro_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            ref_dist=0.05,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        reverse_tiny_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=self.pos,
            pos_b=active_pos,
            ref_dist=0.1,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        pass_distance = nav.dist_1d(active_pos, player_pos)
        opp_dist_to_line = calc.opp_density_to_line(
            self.state, active_pos, player_pos, timestep=self.timestep
        )
        opp_dist_to_active = calc.get_min_opp_distance(
            self.state, active_pos, timestep=self.timestep
        )
        opp_dist_to_active_now = calc.get_min_opp_distance(
            self.state, active_pos, timestep=0
        )
        opp_dist_now = calc.get_min_opp_distance(
            self.state, self.state.team_pos[self.player], timestep=0
        )
        angle_diff = nav.angle_diff(nav.angle(current_dir), nav.angle(desired_dir))
        angle_to_sticky = nav.angle_diff(nav.angle(sticky_dir), nav.angle(desired_dir))

        one_time_kick = True if self.state.will_receive_ball_at >= 3 else False

        prb_success = models.short_pass_success(
            pass_error_diff=self.pass_error_diff,
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_view=self.pos_score_data["view"],
            pos_score_dopp=self.pos_score_data["dopp"],
            pos_score_kopp=self.pos_score_data["kopp"],
            small_cone_angle=small_cone_angle,
            tiny_cone_angle=tiny_cone_angle,
            micro_cone_angle=micro_cone_angle,
            reverse_tiny_cone_angle=reverse_tiny_cone_angle,
            pass_distance=pass_distance,
            opp_dist_to_line=opp_dist_to_line,
            opp_dist_to_active=opp_dist_to_active,
            opp_dist_to_active_now=opp_dist_to_active_now,
            opp_dist_now=opp_dist_now,
            angle_diff=angle_diff,
            angle_to_sticky=angle_to_sticky,
            active_vel=self.state.active_vel,
            player_vel=self.state.team_vel[self.player],
            one_time_kick=one_time_kick,
        )
        self.value *= prb_success ** config.RISK_AVERSION

        pos_offside = calc.position_offside(self.state, player_pos, self.timestep)
        if pos_offside:
            self.value *= 0.1

        self.eval_data = {
            "timestep": self.timestep,
            "kick_countdown": self.state.player_kicked_countdown_timer,
            "pass_error": self.pass_error,
            "pass_error_diff": self.pass_error_diff,
            "pos_offside": pos_offside,
            "small_cone_angle": small_cone_angle,
            "tiny_cone_angle": tiny_cone_angle,
            "micro_cone_angle": micro_cone_angle,
            "reverse_tiny_cone_angle": reverse_tiny_cone_angle,
            "pass_distance": pass_distance,
            "opp_dist_to_line": opp_dist_to_line,
            "opp_dist_to_active": opp_dist_to_active,
            "opp_dist_to_active_now": opp_dist_to_active_now,
            "opp_dist_now": opp_dist_now,
            "angle_diff": angle_diff,
            "angle_to_sticky": angle_to_sticky,
            "active_vel": self.state.active_vel,
            "player_vel": self.state.team_vel[self.player],
            "one_time_kick": one_time_kick,
            "prb_success": prb_success,
            "kick_hold_length": self.kick_hold_length,
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
            if self.state.kick_hold_length_so_far > 0:
                if self.kick_hold_length > self.state.kick_hold_length_so_far:
                    return Action.ShortPass
            if self.action_direction not in self.state.sticky_actions:
                return self.action_direction
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.ReleaseDribble
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
        kick_hold_length,
        error=None,
        error_diff=None,
    ):
        self.player = player
        self.action_direction = action_direction
        self.pass_error = error
        self.pass_error_diff = error_diff
        self.kick_hold_length = kick_hold_length
        self.follow_through = False
        super().__init__(timestep=timestep)

    def evaluate(self):

        self.value = self.position_score()
        self.value += config.LONG_PASS_BONUS

        shoot_plan = Shoot(timestep=self.timestep, player=self.player)
        self.shoot_value = shoot_plan.value * 0.15
        self.value += self.shoot_value

        active_pos = self.state.active_pred[self.timestep]
        player_pos = self.state.team_pred[self.player, self.timestep]

        current_dir = self.state.active_dir
        sticky_dir = nav.action_to_vector_map[self.action_direction]
        desired_dir = (self.pos - self.state.active_pos) / (self.timestep + 1)

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
        tiny_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            ref_dist=0.1,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        micro_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=active_pos,
            pos_b=self.pos,
            ref_dist=0.05,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        reverse_tiny_cone_angle = nav.min_opp_angle(
            state=self.state,
            pos_a=self.pos,
            pos_b=active_pos,
            ref_dist=0.1,
            ref_offset=0.0,
            timestep=self.timestep,
        )
        pass_distance = nav.dist_1d(active_pos, player_pos)
        opp_dist_to_line = calc.opp_density_to_line(
            self.state, active_pos, self.pos, timestep=self.timestep
        )
        opp_dist_to_active = calc.get_min_opp_distance(
            self.state, active_pos, timestep=self.timestep
        )
        opp_dist_to_active_now = calc.get_min_opp_distance(
            self.state, active_pos, timestep=0
        )
        opp_dist_now = calc.get_min_opp_distance(
            self.state, self.state.team_pos[self.player], timestep=0
        )
        angle_diff = nav.angle_diff(nav.angle(current_dir), nav.angle(desired_dir))
        angle_to_sticky = nav.angle_diff(nav.angle(sticky_dir), nav.angle(desired_dir))

        one_time_kick = True if self.state.will_receive_ball_at >= 3 else False

        prb_success = models.long_pass_success(
            pass_error_diff=self.pass_error_diff,
            pos_score_posx=self.pos_score_data["posx"],
            pos_score_dnet=self.pos_score_data["dnet"],
            pos_score_view=self.pos_score_data["view"],
            pos_score_dopp=self.pos_score_data["dopp"],
            pos_score_kopp=self.pos_score_data["kopp"],
            small_cone_angle=small_cone_angle,
            forward_cone_angle=forward_cone_angle,
            tiny_cone_angle=tiny_cone_angle,
            pass_distance=pass_distance,
            opp_dist_to_line=opp_dist_to_line,
            opp_dist_to_active=opp_dist_to_active,
            opp_dist_to_active_now=opp_dist_to_active_now,
            opp_dist_now=opp_dist_now,
            angle_diff=angle_diff,
            angle_to_sticky=angle_to_sticky,
            active_vel=self.state.active_vel,
            player_vel=self.state.team_vel[self.player],
            one_time_kick=one_time_kick,
        )
        self.value *= prb_success ** config.RISK_AVERSION

        pos_offside = calc.position_offside(self.state, player_pos, self.timestep)
        if pos_offside:
            self.value *= 0.1

        self.eval_data = {
            "timestep": self.timestep,
            "kick_countdown": self.state.player_kicked_countdown_timer,
            "pass_error": self.pass_error,
            "pass_error_diff": self.pass_error_diff,
            "pos_offside": pos_offside,
            "small_cone_angle": small_cone_angle,
            "forward_cone_angle": forward_cone_angle,
            "tiny_cone_angle": tiny_cone_angle,
            "micro_cone_angle": micro_cone_angle,
            "reverse_tiny_cone_angle": reverse_tiny_cone_angle,
            "pass_distance": pass_distance,
            "opp_dist_to_line": opp_dist_to_line,
            "opp_dist_to_active": opp_dist_to_active,
            "opp_dist_to_active_now": opp_dist_to_active_now,
            "opp_dist_now": opp_dist_now,
            "angle_diff": angle_diff,
            "angle_to_sticky": angle_to_sticky,
            "active_vel": self.state.active_vel,
            "player_vel": self.state.team_vel[self.player],
            "one_time_kick": one_time_kick,
            "prb_success": prb_success,
            "kick_hold_length": self.kick_hold_length,
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
            if self.state.kick_hold_length_so_far > 0:
                if self.kick_hold_length > self.state.kick_hold_length_so_far:
                    return Action.LongPass
            if self.action_direction not in self.state.sticky_actions:
                return self.action_direction
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.ReleaseDribble
        else:
            return Action.LongPass


class HighPass(Plan):
    name = "HIGH_PASS"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 5
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        # tall skinny rectangle by opp goal
        if self.state.active_pos[0] > 0.91:
            if abs(self.state.active_pos[1]) > 0.1:
                self.value = 52.0
        # shorter wider rectangle
        if self.state.active_pos[0] > 0.83:
            if abs(self.state.active_pos[1]) > 0.2:
                self.value = 52.0
        # far back and alone in own half
        if self.state.active_pos[0] < -0.5:
            if self.state.min_opp_distance[self.state.active_idx] > 0.08:
                self.value = 10.0

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
        if self.follow_through:
            if self.state.kick_hold_length_so_far > 0:
                if self.kick_hold_length > self.state.kick_hold_length_so_far:
                    return Action.HighPass
            if desired_act not in self.state.sticky_actions:
                return desired_act
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.ReleaseDribble
        else:
            return Action.HighPass


class PlayerEight(Plan):
    name = "PLAYER_EIGHT"

    def __init__(self):
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.active_idx == 8:
            if self.state.active_pos[0] > 0.25:
                if self.state.active_pos[1] < -0.2:
                    self.value = 47.0

    def get_position(self):
        return np.array([0.95, -0.35])

    def get_action(self):
        subplan = Move(self.pos)
        return subplan.get_action()


class GoToBall(Plan):
    name = "GO_TO_BALL"

    def __init__(self):
        super().__init__()

    def evaluate(self):
        if self.state.will_receive_ball_at <= 6:
            self.value = 2.0
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
        self.kick_hold_length = 2
        super().__init__(timestep=timestep)

    def evaluate(self):

        if self.player is None:
            self.player = self.state.active_idx

        pos = self.state.team_pred[self.player, self.timestep]
        view_of_net = calc.get_view_of_net(self.state, pos, timestep=self.timestep)
        distance_to_net = calc.get_distance_to_net(pos)
        shooter_kopp = calc.get_opp_kernel_density(self.state, pos, self.timestep)

        if config.USE_HARDCODED_SHOT_VALUE:
            prb_success = -0.99
            opp_goal_dist = calc.opp_distance_to_line(
                self.state, pos, nav.opp_goal, timestep=self.timestep
            )
            self.value = (
                5.0 * (distance_to_net < 0.5)
                + view_of_net * 100.0
                + 10.0 * (distance_to_net < 0.3)
                + 100.0 * opp_goal_dist
            )
        else:
            self.value = 570.0 if distance_to_net < 0.50 else 0.0  # TODO: tune this
            prb_success = models.shot_success(
                view_of_net=view_of_net,
                distance_to_net=distance_to_net,
                posx=pos[0],
                posy=pos[1],
                shooter_kopp=shooter_kopp,
            )
            self.value *= prb_success
        # override if you're really right in front of the net
        if pos[0] > 0.8:
            if abs(pos[1]) < 0.1:
                self.value = 100.0

        self.eval_data = {
            "pos": list(pos.round(4)),
            "view_of_net": view_of_net,
            "distance_to_net": distance_to_net,
            "shooter_kopp": shooter_kopp,
            "prb_success": prb_success,
        }

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
        if self.follow_through:
            if self.state.kick_hold_length_so_far > 0:
                if self.kick_hold_length > self.state.kick_hold_length_so_far:
                    return Action.Shot
            if desired_act not in self.state.sticky_actions:
                return desired_act
            elif self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.ReleaseDribble
        else:
            self.state.write_to_log(
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
            if self.state.opp_will_get_ball_at <= 7:
                will_receive_ball_at = self.state.will_receive_ball_at
                if np.isnan(will_receive_ball_at):
                    will_receive_ball_at = 99
                if self.state.opp_will_get_ball_at < will_receive_ball_at:
                    self.value = 55.0

    def get_position(self):
        opp = self.state.opp_that_will_get_ball or self.state.opp_closest_to_ball
        self.opp = opp
        for t in range(1, self.state.n_step_pred + 1):
            opp_pos = self.state.opp_pred[opp, t]
            opp_dir = self.state.opp_dir[opp]
            if nav.dist_1d(opp_pos, self.state.active_pos) < (0.01 * t):  # old: 0.0125
                vec_to_goal = nav.own_goal - opp_pos
                proj = np.dot(opp_dir, vec_to_goal) / np.linalg.norm(vec_to_goal)
                return opp_pos + vec_to_goal * max(proj * 5.0, 0.005)
        return self.state.opp_pred[opp, -1]

    def get_action(self):
        player_dnet_6 = nav.dist_1d(
            self.state.team_pred[self.state.active_idx, 6], nav.own_goal
        )
        opp_dnet_6 = nav.dist_1d(self.state.opp_pred[self.opp, 6], nav.own_goal)
        if player_dnet_6 > opp_dnet_6:
            if player_dnet_6 - opp_dnet_6 > 0.06:
                subplan = Move(nav.own_goal)
                return subplan.get_action()
        player_dnet_3 = nav.dist_1d(
            self.state.team_pred[self.state.active_idx, 3], nav.own_goal
        )
        opp_dnet_3 = nav.dist_1d(self.state.opp_pred[self.opp, 3], nav.own_goal)
        if opp_dnet_3 < player_dnet_3:
            subplan = Move(nav.own_goal)
            return subplan.get_action()
        subplan = Move(self.pos)
        return subplan.get_action()


class GoalieKick(Plan):
    name = "GOALIE_KICK"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 3
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.active_idx == 0:
            if self.state.will_receive_ball_at <= 6:
                self.value = 200.0
        if nav.dist_1d(self.state.team_pos[0], self.state.ball_pos[:2]) < 0.02:
            if self.state.ball_pos[2] > 0.8:
                self.value = 199.0

    def get_position(self):
        return nav.origin

    def get_action(self):
        desired_act = Action.TopRight
        if self.follow_through:
            if (
                (self.state.kick_hold_length_so_far > 0)
                and (self.kick_hold_length > self.state.kick_hold_length_so_far)
                and not (self.state.will_receive_ball_at <= 2)
            ):
                return Action.HighPass
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                return Action.ReleaseDribble
        return Action.HighPass


class CornerKick(Plan):
    name = "CORNER_KICK"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 3
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.game_mode == GameMode.Corner:
            self.value = 200.0

    def get_position(self):
        return nav.opp_goal

    def get_action(self):
        desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
        if self.follow_through:
            if self.state.kick_hold_length_so_far > 0:
                if self.kick_hold_length > self.state.kick_hold_length_so_far:
                    return Action.HighPass
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                return Action.ReleaseDribble
        else:
            return Action.HighPass


class FreeKick(Plan):
    name = "FREE_KICK"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 1
        super().__init__()

    def evaluate(self):
        self.value = 0.0
        if self.state.game_mode == GameMode.FreeKick:
            self.value = 200.0

    def get_position(self):
        return nav.origin

    def get_action(self):
        desired_act = Action.TopRight
        if self.follow_through:
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                return Action.ReleaseDribble
        else:
            # if desired_act not in self.state.sticky_actions:
            #     return desired_act
            self.state.write_to_log({"type": "FREE_KICK_ATTEMPT"})
            return Action.ShortPass


class Breakaway(Plan):
    name = "BREAKAWAY"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 1
        self.is_decision_point = False
        super().__init__()

    def evaluate(self):

        self.value = 3.0

        if self.state.active_pos[0] > 0.0:
            if self.state.active_pos[0] < 0.8:
                active_dist_to_net = nav.dist_1d(self.state.active_pos, nav.opp_goal)
                opp_dist_to_net = nav.dist_2d(self.state.opp_pos, nav.opp_goal)
                if (active_dist_to_net > opp_dist_to_net).sum() <= 1:
                    if state.left_has_ball:
                        self.value = 99.0
                    else:
                        self.value = 49.0
                else:
                    self.value = 6.0
            else:
                self.value = 5.0

        if state.left_has_ball:
            active_pos = self.state.active_pred[6]
            goalie_pos = self.state.opp_pred[0, 6]
            dist_to_goalie = nav.dist_1d(active_pos, goalie_pos)
            if dist_to_goalie <= 0.21:
                if self.state.player_kicked_countdown_timer == 0:
                    self.is_decision_point = True

    def get_position(self):
        return nav.opp_goal

    def get_action(self):

        if self.follow_through:
            if self.state.is_sprinting:
                return Action.ReleaseSprint
            else:
                return Action.ReleaseDribble

        if self.is_decision_point:
            desired_act = (
                Action.TopRight if self.state.active_pos[1] > 0 else Action.BottomRight
            )
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                self.state.write_to_log({"type": "BREAKAWAY_SHOT_ATTEMPT"})
                return Action.Shot

        subplan = Move(self.pos)
        return subplan.get_action()


class Kickoff(Plan):
    name = "KICKOFF"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 1
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
                return Action.ReleaseDribble
        else:
            return Action.ShortPass


class FinalDefense(Plan):
    name = "FINAL_DEFENSE"

    def evaluate(self):

        self.value = 0.0

        if self.state.left_has_ball:
            return None

        if self.state.will_receive_ball_at <= self.state.opp_will_get_ball_at:
            return None

        team_distance_to_goal = nav.dist_2d(self.state.team_pred[:, 4], nav.own_goal)
        player_closest_to_goal = np.argmin(team_distance_to_goal[1:]) + 1
        player_distance_to_goal = team_distance_to_goal[player_closest_to_goal]
        if player_closest_to_goal == self.state.active_idx:
            opp_distance_to_goal = nav.dist_2d(self.state.opp_pred[:, 4], nav.own_goal)
            if min(opp_distance_to_goal) - 0.01 < player_distance_to_goal:
                self.value = 50.5  # higher than chase ball, lower than chase player

    def get_position(self):
        return nav.own_goal

    def get_action(self):
        subplan = Move(self.pos)
        return subplan.get_action()


class PassBack(Plan):
    name = "PASS_BACK"

    def __init__(self):
        self.follow_through = False
        self.kick_hold_length = 1
        super().__init__()

    def evaluate(self):

        self.value = 1.0

        if self.state.active_idx == 0:
            return None
        if self.state.player_kicked_countdown_timer > 0:
            return None

        active_distance_to_goalie = nav.dist_1d(
            self.state.active_pos, self.state.team_pos[0]
        )
        opp_distance_to_goalie = min(
            nav.dist_2d(self.state.opp_pos, self.state.team_pos[0])
        )
        if active_distance_to_goalie < opp_distance_to_goalie:
            if active_distance_to_goalie > 0.12:
                if active_distance_to_goalie < 1.0:
                    closest_opp = np.argmin(
                        self.state.opp_distance_matrix[self.state.active_idx]
                    )
                    closest_opp_pos = self.state.opp_pos[closest_opp]
                    closest_opp_distance = self.state.opp_distance_matrix[
                        self.state.active_idx, closest_opp
                    ]
                    if closest_opp_distance < 0.15:
                        if closest_opp_pos[0] > self.state.active_pos[0]:
                            self.value = 70.0

    def get_position(self):
        return self.state.team_pos[0]

    def get_action(self):
        if self.follow_through:
            desired_act = nav.get_action_direction(self.state.active_pos, self.pos)
            if desired_act not in self.state.sticky_actions:
                return desired_act
            else:
                return Action.ReleaseDribble
        return Action.ShortPass
