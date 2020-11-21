import numpy as np

from kaggle_environments.envs.football.helpers import Action

# PSA: all angles in degrees!

# TODO: do this with a list and mod math instead?
deg_to_action_map = {
    0: Action.Right,
    45: Action.BottomRight,
    90: Action.Bottom,
    135: Action.BottomLeft,
    180: Action.Left,
    -180: Action.Left,
    -135: Action.TopLeft,
    -90: Action.Top,
    -45: Action.TopRight,
}

half_2 = 0.5 ** 0.5

# all vectors in the following have norm 1
action_to_vector_map = {
    Action.Right: np.array([1.0, 0.0]),
    Action.BottomRight: np.array([half_2, half_2]),
    Action.Bottom: np.array([0.0, 1.0]),
    Action.BottomLeft: np.array([-half_2, half_2]),
    Action.Left: np.array([-1.0, 0.0]),
    Action.TopLeft: np.array([-half_2, -half_2]),
    Action.Top: np.array([0.0, -1.0]),
    Action.TopRight: np.array([half_2, -half_2]),
}

origin = np.array([0.0, 0.0])
own_goal = np.array([-1.0, 0.0])
opp_goal = np.array([1.0, 0.0])
opp_post_a = np.array([1.0, 0.044])
opp_post_b = np.array([1.0, -0.044])
invalid = np.array([99.0, 99.0])


def normalise_1d(xy):
    """Scales 1-d vector by its norm"""
    denom = np.linalg.norm(xy)
    return xy / denom if denom else 0.0


def normalise_2d(xy):
    """Scales each row of matrix by its norm"""
    return xy / np.linalg.norm(xy, axis=1)


def dist_1d(xy_a, xy_b):
    """Distance between two 2-d positions"""
    return np.linalg.norm(xy_a - xy_b)


def dist_2d(xy_a, xy_b):
    """Distance between two 2-d positions"""
    return np.linalg.norm(xy_a - xy_b, axis=1)


def angle(xy):
    """Returns the angle(s) of 2-dimensional vector(s) in degrees"""
    # xy may have 1 or 2 dimensions, assume x/y is indexed by final dim
    return np.arctan2(xy[..., 1], xy[..., 0]) * 180 / np.pi


def angle_diff(angle_a, angle_b):
    """The smallest absolute angle between two angles given in degrees"""
    # arguments can be either scalars or compatible vectors
    diff = angle_a - angle_b
    return np.absolute((diff + 180) % 360 - 180)


def snap_to_45(angle):
    """Round angle to the nearest 45 degrees"""
    return int(round(angle / 45) * 45)


def outside_pitch(pos, tolerance=0.04):
    """Is position within the pitch boundary?"""
    if pos[0] < (-1 + tolerance):
        return 1
    if pos[0] > (1 - tolerance):
        return 1
    if pos[1] < (-0.42 + tolerance):
        return 1
    if pos[1] > (0.42 - tolerance):
        return 1
    return 0


def get_action_direction(pos_a, pos_b):
    """Closest action to move from a --> b"""
    deg = angle(pos_b - pos_a)
    return deg_to_action_map[snap_to_45(deg)]


def get_action_direction_with_dir(pos_a, pos_b, dir_a):
    """Best action to move from a --> b given current direction of a.

    Probably this will be most useful while sprinting
    """
    vec_to_b = normalise_1d(pos_b - pos_a)
    vec_target = vec_to_b - normalise_1d(dir_a) / 2
    deg = angle(vec_target)
    return deg_to_action_map[snap_to_45(deg)]


def pass_error(team_pos, active_idx, action):
    active_pos = team_pos[active_idx]
    pass_vec = action_to_vector_map[action] * 0.15
    pass_pos = active_pos + pass_vec
    pass_dist = dist_2d(team_pos, pass_pos)
    pass_angle = angle_diff(angle(pass_pos - active_pos), angle(team_pos - active_pos))
    pass_error = (pass_dist * 100.0) ** 0.7 + pass_angle * 0.64
    pass_error[active_idx] = 99_999.0
    return pass_error


# # from: https://github.com/google-research/football/blob/master/third_party/gfootball_engine/src/onthepitch/AIsupport/AIfunctions.cpp#L1093  # noqa
# def pass_error_2(team_pos, team_dir, active_idx, action, pass_type):
#     active_pos = team_pos[active_idx]
#     action_dir = action_to_vector_map[action]
#     team_dist = dist_2d(team_pos, active_pos)

#     GAUGE_MS = 200  # no idea how to set this at the moment
#     INPUT_DIRECTION_SCALING = 0.01
#     DISTANCE_RATING_SCALING = 0.1
#     TOTAL_SCALING = 30.0

#     gauge_factor = np.clip((GAUGE_MS - 60) / (1000 - 60), 0.0, 1.0)

#     if pass_type == "SHORT_PASS":
#         input_power = np.clip(gauge_factor ** 0.7, 0.01, 1.0)
#     elif pass_type == "LONG_PASS":
#         input_power = np.clip(gauge_factor ** 0.65, 0.01, 1.0)
#     elif pass_type == "HIGH_PASS":
#         input_power = np.clip(gauge_factor ** 0.55, 0.01, 1.0)
#     else:
#         raise ValueError("unknown pass_type")

#     manual_target = active_pos + action_dir * INPUT_DIRECTION_SCALING * (
#         input_power * 60.0
#     ).clip(min=1.0, max=100.0)

#     pass_duration = 0.3 + team_dist * 0.05
#     target_pos = team_pos + team_dir * pass_duration.reshape(-1, 1)
#     distance_rating = (
#         dist_2d(target_pos, manual_target).clip(min=0.0, max=70.0) ** 0.8 * 0.8
#     ) * DISTANCE_RATING_SCALING

#     radians_vs_action = np.radians(
#         angle_diff(angle(target_pos - team_pos), angle(action_dir))
#     )
#     angle_rating = np.abs(radians_vs_action) / np.pi

#     total_rating = distance_rating + angle_rating
#     total_rating *= TOTAL_SCALING
#     total_rating[active_idx] = 99_999.0

#     return total_rating


def who_will_receive_pass(team_pos, active_idx, action):
    errors = pass_error(team_pos, active_idx, action)
    predicted_player = np.argmin(errors)
    lowest_error, second_lowest_error = sorted(errors)[:2]
    error_diff = abs(lowest_error - second_lowest_error)
    return predicted_player, lowest_error, error_diff


# def who_will_receive_pass_2(team_pos, team_dir, active_idx, action, pass_type):
#     errors = pass_error_2(team_pos, team_dir, active_idx, action, pass_type)
#     predicted_player = np.argmin(errors)
#     lowest_error, second_lowest_error = sorted(errors)[:2]
#     error_diff = abs(lowest_error - second_lowest_error)
#     return predicted_player, lowest_error, error_diff


def min_opp_angle(
    state,
    pos_a,
    pos_b,
    ref_dist=None,
    ref_offset=-0.02,
    timestep=0,
):

    target_vec = normalise_1d(pos_b - pos_a)
    target_angle = angle(target_vec)
    reference_pos = pos_a + target_vec * ref_offset
    distance_window = ref_dist or dist_1d(pos_a, pos_b) + 0.02
    opp_distances = dist_2d(pos_a, state.opp_pred[:, timestep])

    min_angle = 200.0
    for opp_idx in np.where(opp_distances < distance_window)[0]:

        opp_pos = state.opp_pred[opp_idx, timestep]
        opp_angle = angle(opp_pos - reference_pos)
        tmp_angle = angle_diff(target_angle, opp_angle)

        if tmp_angle < min_angle:
            min_angle = tmp_angle

    return min_angle


def opp_in_cone(
    state,
    pos_a,
    pos_b,
    ref_dist=None,
    ref_angle=25,
    ref_offset=-0.02,
    timestep=0,
):
    min_angle = min_opp_angle(
        state=state,
        pos_a=pos_a,
        pos_b=pos_b,
        ref_dist=ref_dist,
        ref_offset=ref_offset,
        timestep=timestep,
    )
    if min_angle < ref_angle:
        return True
    return False


# from: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def dist_from_point_to_line_segment(point_pos, line_pos_a, line_pos_b):
    line_length = dist_1d(line_pos_a, line_pos_b)
    if line_length == 0:
        return dist_1d(line_pos_a, point_pos)
    t = np.dot(point_pos - line_pos_a, line_pos_b - line_pos_a) / (line_length ** 2)
    t = max(0, min(1, t))
    projection = line_pos_a + t * (line_pos_b - line_pos_a)
    return dist_1d(point_pos, projection)


if __name__ == "__main__":

    # tests:

    a = np.array([0, 0])
    b = np.array([-0, 1.5])
    print(get_action_direction(a, b))

    pos_a = np.array([0, 0])
    pos_b = np.array([1, 0])
    dir_a = np.array([1, -1])
    print(get_action_direction_with_dir(pos_a, pos_b, dir_a))

    pos_a = np.array([0, 0])
    pos_b = np.array([1, 0])
    dir_a = np.array([-1, 0])
    print(get_action_direction_with_dir(pos_a, pos_b, dir_a))

    pos_a = np.array([0, 0])
    pos_b = np.array([1, 0])
    dir_a = np.array([-1, -1])
    print(get_action_direction_with_dir(pos_a, pos_b, dir_a))

    pos_a = np.array([0, 0])
    pos_b = np.array([1, 1])
    dir_a = np.array([1, 0])
    print(get_action_direction_with_dir(pos_a, pos_b, dir_a))
