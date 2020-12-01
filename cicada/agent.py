import time
from copy import deepcopy

import numpy as np
from kaggle_environments.envs.football.helpers import Action

from cicada.utils import data
from cicada.utils import plans
from cicada.utils import config
from cicada.utils import navigation as nav


class Agent:
    def __init__(self, noise_sd=0.0):
        self.state = None
        self.noise_sd = noise_sd
        self.previous_plan = None

    def action_wrapper(self, obs):

        self.start_time = time.time()

        obs = deepcopy(obs)
        obs = data.clean_observation(obs)

        action = self.action(obs)

        return [action.value]

    def action(self, obs):

        if self.state is None:
            self.state = data.State(obs)
            plans.state = self.state  # set at module level
        else:
            self.state.start_of_turn_update(obs)

        state = self.state  # to save myself writing `self` all the time

        # log the active pos score so we can calibrate it later
        if state.active_has_ball:
            active_pos_score_data = plans.Move(state.active_pos).pos_score_data
            active_pos_score_data["type"] = "ACTIVE_POS_SCORE"
            state.write_to_log(active_pos_score_data)

        breakaway_decision_point = False

        in_follow_through = False
        if state.player_kicked_countdown_timer > 0:
            if state.active_idx == state.player_kicked:
                in_follow_through = True

        # figure out all the plans we could do
        potential_plans = list()

        potential_plans.append(plans.GoToBall())
        potential_plans.append(plans.ChasePlayer())
        potential_plans.append(plans.Kickoff())
        potential_plans.append(plans.CornerKick())
        potential_plans.append(plans.FreeKick())
        # potential_plans.append(plans.FinalDefense())
        potential_plans.append(plans.GoalieKick())

        if (not state.right_has_ball) and (
            state.active_has_ball
            or (state.will_receive_ball_at <= 3)
            or (
                (state.will_receive_ball_at <= 6)
                and (
                    (state.will_receive_ball_at < state.opp_will_get_ball_at)
                    or np.isnan(state.opp_will_get_ball_at)
                )
            )
        ):

            if state.will_receive_ball_at > 2:
                timestep = state.will_receive_ball_at + 2
            else:
                timestep = config.KICK_TIMESTEP

            potential_plans.append(plans.PassBack())
            potential_plans.append(plans.HighPass())
            potential_plans.append(plans.PlayerEight())
            potential_plans.append(plans.Shoot(timestep=timestep))

            breakaway = plans.Breakaway()
            breakaway_decision_point = (
                breakaway.is_decision_point and (breakaway.value > 60)
            )
            potential_plans.append(breakaway)

            players_to_pass_to = dict()  # player: action, error, error_diff

            for action in nav.action_to_vector_map.keys():

                for kick_hold_length in (1, 2, 3):

                    if state.ball_pos[0] > 0.0:
                        if kick_hold_length == 3:
                            continue
                        if state.ball_pos[0] > 0.5:
                            if kick_hold_length == 2:
                                continue

                    if in_follow_through:
                        if state.kick_hold_length_so_far > kick_hold_length:
                            continue
                        if not state.kicked_last_turn:
                            if state.kick_hold_length_so_far != kick_hold_length:
                                continue

                    player, error, error_diff = nav.who_will_receive_pass(
                        state.team_pos, state.active_idx, action, kick_hold_length
                    )
                    _, _, _, curr_diff = players_to_pass_to.get(
                        player, [None, None, 999.9, 0.0]
                    )

                    if error < 25.0:  # if error worse than this, targeting likely poor
                        if error_diff > curr_diff:
                            players_to_pass_to[player] = [
                                action,
                                kick_hold_length,
                                error,
                                error_diff,
                            ]

            for player, (
                action,
                kick_hold_length,
                error,
                error_diff,
            ) in players_to_pass_to.items():
                if kick_hold_length <= 2:
                    potential_plans.append(
                        plans.ShortPass(
                            player,
                            action,
                            timestep=timestep,
                            kick_hold_length=kick_hold_length,
                            error=error,
                            error_diff=error_diff,
                        )
                    )
                potential_plans.append(
                    plans.LongPass(
                        player,
                        action,
                        timestep=timestep,
                        kick_hold_length=kick_hold_length,
                        error=error,
                        error_diff=error_diff,
                    )
                )

            move_timestep = config.MOVE_WITH_BALL_TIMESTEP
            if state.will_receive_ball_at > 0:
                move_timestep += state.will_receive_ball_at
            for action, _ in nav.action_to_vector_map.items():
                potential_plans.append(
                    plans.MoveWithBall(action, timestep=move_timestep)
                )

        # boost follow-through plans
        if in_follow_through:
            time_since_kick = abs(
                state.player_kicked_countdown_timer - config.KICK_COUNTDOWN_TIMER_START
            )
            for plan in potential_plans:
                if isinstance(plan, state.follow_through_plan):
                    plan.follow_through = True
                    plan.timestep = max(0, plan.timestep - time_since_kick)
                    plan.evaluate()  # reevaluate with new timestep
                    plan.value += 100 * (plan.value > 0)
                    plan.name += "_FT"

        if config.CONSIDER_BREAKAWAY_PASS:
            if breakaway_decision_point:
                print(f"breakaway decision point at step {state.step}")
                for plan in potential_plans:
                    # if isinstance(plan, (plans.ShortPass, plans.LongPass)):
                    if isinstance(plan, plans.LongPass):
                        player_pos = state.team_pos[plan.player]
                        if player_pos[0] > state.active_pos[0] - 0.17:
                            if plan.action_direction in (
                                Action.BottomRight,
                                Action.Bottom,
                                Action.Top,
                                Action.TopRight,
                            ):
                                plan.value += 75 * (plan.value > 0)
                                plan.value += plan.pos_score_data.get("view", 0) * 10
                                plan.name += "_BR"

        # randomise values
        for plan in potential_plans:
            if plan.randomisable:
                plan.rand = np.random.normal(loc=0.0, scale=self.noise_sd)
                plan.value += plan.rand

        # boost previous plan
        for plan in potential_plans:
            if plan == self.previous_plan:
                plan.value += 0.8
                plan.name += "*"

        # pick the best plan
        potential_plans = sorted(potential_plans, key=lambda x: -x.value)
        plan = potential_plans[0]
        self.previous_plan = plan

        action = plan.get_action()

        if action is None:
            raise ValueError(f"Plan {plan} returned 'None' as action :(")

        # if we've kicked the ball, activate follow-through
        if action in [Action.ShortPass, Action.LongPass, Action.HighPass, Action.Shot]:

            state.player_kicked = state.active_idx
            state.player_kicked_countdown_timer = config.KICK_COUNTDOWN_TIMER_START
            state.follow_through_plan = type(plan)

            if state.kicked_last_turn:
                state.kick_hold_length_so_far += 1
            else:
                state.kick_hold_length_so_far = 1

            state.kicked_last_turn = True

            if not state.left_has_ball:
                if not np.isnan(state.will_receive_ball_at):
                    state.write_to_log(
                        {
                            "type": "ONE_TIME_KICK_ATTEMPT",
                            "will_receive_ball_at": state.will_receive_ball_at,
                        }
                    )
        else:
            state.kicked_last_turn = False
            if not in_follow_through:
                state.kick_hold_length_so_far = 0

        for plan in potential_plans:
            plan.end_of_turn_update()

        if action == state.action:
            state.action_repeat_count += 1
        else:
            state.action_repeat_count = 0
        state.action = action
        state.elapsed_time = time.time() - self.start_time
        state.end_of_turn_update()

        return action


if __name__ == "__main__":

    # for testing/debugging

    agent = Agent()

    from kaggle_environments import make

    env = make(
        "football",
        configuration={
            "save_video": False,
            "scenario_name": "11_vs_11_kaggle",
            "episodeSteps": 500,
        },
    )

    env.run([agent.action_wrapper, "cicada/simple_rules_based_agent.py"])

    trainer = env.train([None, "cicada/simple_rules_based_agent.py"])
    obs = trainer.reset()
    while not env.done:
        action = agent.action_wrapper(obs)
        obs, reward, done, info = trainer.step(action)

    print("done")
