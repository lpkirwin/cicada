from kaggle_environments.envs.football.helpers import Action, human_readable_agent


@human_readable_agent
def agent(obs):
    # Make sure player is running.
    if Action.Sprint not in obs["sticky_actions"]:
        return Action.Sprint
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs["left_team"][obs["active"]]
    # Does the player we control have the ball?
    if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_team"] == 0:
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        # Shot if we are 'close' to the goal (based on 'x' coordinate).
        if controlled_player_pos[0] > 0.5:
            return Action.Shot
        # pass if not above halfline
        if controlled_player_pos[0] < -0.1:
            if Action.Right in obs["sticky_actions"]:
                return Action.HighPass
        #         # dribble just above halfline
        #         if controlled_player_pos[0] > 0 and controlled_player_pos[0] < 0.3:
        #             return Action.Dribble
        return Action.Idle
    else:
        # Run towards the ball.
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
