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

        # figure out all the plans we could do
        potential_plans = list()

        potential_plans.append(plans.GoToBall())
        potential_plans.append(plans.ChasePlayer())
        potential_plans.append(plans.GoalieKick())
        potential_plans.append(plans.Kickoff())
        potential_plans.append(plans.CornerKick())
        potential_plans.append(plans.FreeKick())

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

            extra_timestep = state.will_receive_ball_at

            potential_plans.append(plans.Breakaway())
            potential_plans.append(plans.HighPass())
            potential_plans.append(
                plans.Shoot(timestep=config.SHOOT_TIMESTEP + extra_timestep)
            )

            players_to_pass_to = dict()  # player: action, error, error_diff
            for action in nav.action_to_vector_map.keys():
                player, error, error_diff = nav.who_will_receive_pass(
                    state.team_pos, state.active_idx, action
                )
                curr_act, curr_err, curr_diff = players_to_pass_to.get(
                    player, [None, 999.9, 0.0]
                )
                if error < 25.0:  # if error worse than this, targeting likely poor
                    if error_diff > curr_diff:  # replace
                        players_to_pass_to[player] = [action, error, error_diff]

            for player, (action, error, error_diff) in players_to_pass_to.items():
                potential_plans.append(
                    plans.ShortPass(
                        player,
                        action,
                        timestep=config.PASS_TIMESTEP + extra_timestep,
                        error=error,
                        error_diff=error_diff,
                    )
                )
                potential_plans.append(
                    plans.LongPass(
                        player,
                        action,
                        timestep=config.PASS_TIMESTEP + extra_timestep,
                        error=error,
                        error_diff=error_diff,
                    )
                )

            for action, _ in nav.action_to_vector_map.items():
                potential_plans.append(
                    plans.MoveWithBall(
                        action,
                        timestep=config.MOVE_WITH_BALL_TIMESTEP + extra_timestep,
                    )
                )

        # boost follow-through plans
        if state.player_kicked_countdown_timer > 0:
            if state.active_idx == state.player_kicked:
                time_since_kick = abs(
                    state.player_kicked_countdown_timer
                    - config.KICK_COUNTDOWN_TIMER_START
                )
                for plan in potential_plans:
                    if isinstance(plan, state.follow_through_plan):
                        plan.follow_through = True
                        plan.timestep = max(0, plan.timestep - time_since_kick)
                        plan.evaluate()  # reevaluate with new timestep
                        plan.value += 100
                        plan.name += "_FT"

        # randomise values
        for plan in potential_plans:
            if plan.randomisable:
                plan.rand = np.random.normal(loc=0.0, scale=self.noise_sd)
                plan.value += plan.rand

        # pick the best plan
        potential_plans = sorted(potential_plans, key=lambda x: -x.value)
        plan = potential_plans[0]

        action = plan.get_action()

        if action is None:
            raise ValueError(f"Plan {plan} returned 'None' as action :(")

        # if we've kicked the ball, activate follow-through
        if action in [Action.ShortPass, Action.LongPass, Action.HighPass, Action.Shot]:
            state.player_kicked = state.active_idx
            state.player_kicked_countdown_timer = config.KICK_COUNTDOWN_TIMER_START
            state.follow_through_plan = type(plan)
            if not state.left_has_ball:
                if not np.isnan(state.will_receive_ball_at):
                    state.write_to_log(
                        {
                            "type": "ONE_TIME_KICK_ATTEMPT",
                            "will_receive_ball_at": state.will_receive_ball_at,
                        }
                    )

        for plan in potential_plans:
            plan.end_of_turn_update()

        state.action = action
        state.elapsed_time = time.time() - self.start_time
        state.end_of_turn_update()

        return action


if __name__ == "__main__":

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
