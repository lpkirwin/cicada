import numpy as np

from . import navigation as nav


def get_view_of_net(state, pos, timestep=0, log_intercepts=False):

    view_of_net = 1.0

    for opp_pos in state.opp_pred[:, timestep, :]:

        if opp_pos[0] <= pos[0]:
            continue

        opp_a = np.array([opp_pos[0], opp_pos[1] + 0.015])
        opp_b = np.array([opp_pos[0], opp_pos[1] - 0.015])
        slope_a = (opp_a[1] - pos[1]) / (opp_a[0] - pos[0])
        slope_b = (opp_b[1] - pos[1]) / (opp_b[0] - pos[0])
        intercept_a = pos[1] + slope_a * (1.0 - pos[0])
        intercept_b = pos[1] + slope_b * (1.0 - pos[0])

        if log_intercepts:  # for future diagnostic plotting
            state.active_intercepts.extend([intercept_a, intercept_b])

        pct_hidden = (
            abs(
                max(-0.044, min(0.044, intercept_a))
                - max(-0.044, min(0.044, intercept_b))
            )
            / 0.088
        )

        view_of_net = max(0, view_of_net - pct_hidden)

    # scale by angle of net in view (decreases with distance
    # and off-centerness)
    post_angle_a = nav.angle(nav.opp_post_a - pos)
    post_angle_b = nav.angle(nav.opp_post_b - pos)
    net_angle = nav.angle_diff(post_angle_a, post_angle_b)
    view_of_net *= net_angle / 45.0

    return view_of_net


def get_distance_to_net(pos):
    return nav.dist_1d(pos, nav.opp_goal)


def get_min_opp_distance(state, pos, timestep=0):
    return min(min(nav.dist_2d(pos, state.opp_pred[:, timestep])), 0.2)


def get_opp_kernel_density(state, pos, timestep=0):
    opp_dists = nav.dist_2d(pos, state.opp_pred[:, timestep]).clip(min=0.01)
    return (1 / opp_dists).sum()


def get_min_team_distance(state, pos, timestep=0):
    return min(min(nav.dist_2d(pos, state.team_pred[:, timestep])), 0.2)  # TODO?


def position_offside(state, pos, timestep=0):
    try:
        opp_second_last_x = sorted(state.opp_pred[:, timestep, 0])[-2]
    except IndexError:
        return 0.0
    ball_x = state.ball_pred[timestep, 0]
    if pos[0] > opp_second_last_x:
        if ball_x < opp_second_last_x:
            return 1.0
    return 0.0


def opp_density_to_line(state, pos_a, pos_b, timestep=0):
    opp_dists = np.array(
        [
            nav.dist_from_point_to_line_segment(opp_pos, pos_a, pos_b)
            for opp_pos in state.opp_pred[:, timestep]
        ]
    ).clip(min=0.01)
    return (1 / opp_dists).sum()
