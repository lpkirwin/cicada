import plotly.graph_objects as go
import numpy as np

# import pandas as pd
# from copy import deepcopy

from . import navigation as nav
from . import data

# from ipywidgets import interact


def print_diagnostics(state, log_step):

    n_cols = 2
    n_rows = 20
    max_width = 105
    col_width = max_width // n_cols

    array = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    eot_rec = data.filter_log_step(log_step, type="END_OF_TURN")[0]

    # left column
    array[0][0] = f"mode: {eot_rec['game_mode']}"
    array[1][0] = f"score: {state.obs['score']}"
    array[2][0] = f"sticky: [{', '.join([act.name for act in state.sticky_actions])}]"
    array[3][0] = f"action: [{eot_rec['action'].name}]"
    array[4][0] = f"ball_owned_team: [{state.obs['ball_owned_team']}]"
    array[5][0] = f"active player: {state.active_idx}"
    array[6][0] = f"active degrees: {round(state.active_deg, 4)}"
    array[7][0] = f"active velocity: {round(state.active_vel, 4)}"
    array[8][0] = f"kick countdown: {eot_rec['kick_countdown']}"
    array[9][0] = f"kick player: {eot_rec['kick_player']}"
    array[10][0] = f"will receive ball at: {eot_rec['will_receive_ball_at']}"
    array[11][0] = f"can receive ball at: {eot_rec['can_receive_ball_at']}"
    array[12][0] = f"will collide with opp at: {eot_rec['will_collide_with_opp_at']}"
    array[13][0] = f"elapsed time: {round(eot_rec['elapsed_time'], 6)}"

    shot_eval = data.filter_log_step(log_step, type="SHOT_EVALUATION")
    if len(shot_eval):
        array[14][0] = f"shot view of net: {round(shot_eval[0]['view_of_net'], 4)}"
        array[15][0] = f"shot dist to net: {round(shot_eval[0]['distance_to_net'], 4)}"

    # right column
    max_plans = n_rows
    n_plans = 0
    for rec in data.filter_log_step(log_step, type="PLAN"):
        # active = " (ACTIVE)" if rec["active"] else ""
        # array[n_plans][1] = f"{rec['plan']}{active} / {rec['value']} / {rec['pos']}"
        text = (
            str(rec["plan"]).ljust(15)
            + "{:.2f}".format(rec["value"]).rjust(7)
            + "{:.2f}".format(rec["pos_score_data"].get("score", 0.0)).rjust(7)
            + "{:.2%}".format(rec["eval_data"].get("prb_success", 0.0)).rjust(8)
            + str(rec["pos"].round(2)).rjust(15)
        )
        array[n_plans][1] = text
        n_plans += 1
        if n_plans == max_plans:
            break

    for i in range(n_rows):
        print("| ", end="")
        for j in range(n_cols):
            string = array[i][j]
            if len(string) > col_width:
                new_string = string[: col_width - 3] + "..."
            else:
                new_string = string + " " * (col_width - len(string))
            print(new_string, end=" | ")
        print()


class Tooltip:
    def __init__(self):
        self.data = dict()
        self.size = 0

    def add(self, **kwargs):
        for key, values in kwargs.items():
            try:
                len(values)
            except TypeError:
                values = [values]
            self.data[key] = values
            self.size = max(self.size, len(values))

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = list()
            self.data[key].append(value)
            self.size = max(self.size, len(self.data[key]))

    def get_text(self):
        text = ["" for _ in range(self.size)]
        for key, values in self.data.items():
            for i, v in enumerate(values):
                text[i] += f"{key}: {v}<br>"
        return text


def get_traces(state, log_step):

    s = state

    eot_rec = data.filter_log_step(log_step, type="END_OF_TURN")[0]

    active_intercepts = eot_rec["active_intercepts"]
    active_intercepts += [np.nan] * (6 - len(active_intercepts))
    active_intercepts = active_intercepts[:6]

    n_targets_to_show = 14
    target_pos = np.ones(shape=(n_targets_to_show, 2)) * -99
    target_tooltip = Tooltip()
    for i, rec in enumerate(data.filter_log_step(log_step, type="PLAN")):
        target_pos[i] = rec["pos"]
        target_tooltip.append(
            plan=rec["plan"],
            action_direction=rec["action_direction"],
            timestep=rec.get("timestep", None),
            pass_error=round(rec["pass_error"], 4),
            pass_error_diff=round(rec["pass_error_diff"], 4),
            pos_score=round(rec["pos_score_data"]["score"], 4),
            prb_success=round(rec["eval_data"].get("prb_success", 0.0), 4),
            shoot_value=round(rec["shoot_value"], 4),
            value=round(rec["value"], 4),
        )
        if i == n_targets_to_show - 1:
            break

    left_tooltip = Tooltip()
    left_tooltip.add(player=list(range(s.team_n_players)))
    left_tooltip.add(pos_x=s.team_pos[:, 0].round(4))
    left_tooltip.add(pos_y=s.team_pos[:, 1].round(4))
    left_tooltip.add(dir_x=s.team_dir[:, 0].round(4))
    left_tooltip.add(dir_y=s.team_dir[:, 1].round(4))
    left_tooltip.add(dist_to_active=nav.dist_2d(s.team_pos, s.active_pos).round(4))
    left_tooltip.add(active_view_of_net=eot_rec["active_view_of_net"].round(4))
    left_tooltip.add(min_opp_distance=eot_rec["min_opp_distance"].round(4))
    left_tooltip.add(sticky_pass_error=eot_rec["sticky_action_pass_error"].round(4))

    right_tooltip = Tooltip()
    right_tooltip.add(player=list(range(s.opp_n_players)))
    right_tooltip.add(pos_x=s.opp_pos[:, 0].round(4))
    right_tooltip.add(pos_y=s.opp_pos[:, 1].round(4))
    right_tooltip.add(dir_x=s.opp_dir[:, 0].round(4))
    right_tooltip.add(dir_y=s.opp_dir[:, 1].round(4))
    right_tooltip.add(dist_to_active=nav.dist_2d(s.opp_pos, s.active_pos).round(4))

    ball_tooltip = Tooltip()
    ball_tooltip.add(pos_x=round(s.ball_pos[0], 4))
    ball_tooltip.add(pos_y=round(s.ball_pos[1], 4))
    ball_tooltip.add(pos_z=round(s.ball_pos[2], 4))
    ball_tooltip.add(dir_x=round(s.ball_dir[0], 4))
    ball_tooltip.add(dir_y=round(s.ball_dir[1], 4))
    ball_tooltip.add(dir_z=round(s.ball_dir[2], 4))
    ball_tooltip.add(rot_x=round(s.ball_rot[0], 4))
    ball_tooltip.add(rot_y=round(s.ball_rot[1], 4))
    ball_tooltip.add(rot_z=round(s.ball_rot[2], 4))

    traces = list()

    # targets!
    traces.append(
        go.Scatter(
            name="target",
            x=target_pos[:, 0],
            y=target_pos[:, 1],
            mode="markers",
            marker=dict(
                color="darkorange",
                size=30,
                opacity=0.35,
                symbol=["circle"] + ["circle-open"] * (n_targets_to_show - 1),
                # symbol="circle",
            ),
            text=target_tooltip.get_text(),
            hoverinfo="text",
        )
    )

    # ball predictions
    traces.append(
        go.Scatter(
            name="ball_pred",
            x=eot_rec["ball_pred"][:, 0],
            y=eot_rec["ball_pred"][:, 1],
            mode="markers",
            fillcolor=None,
            marker=dict(
                color="white",
                size=2,
                opacity=0.3,
                symbol="circle-open",
            ),
            text="",
            hoverinfo="skip",
        )
    )

    # active predictions
    traces.append(
        go.Scatter(
            name="active_pred",
            x=eot_rec["active_pred"][:, 0],
            y=eot_rec["active_pred"][:, 1],
            mode="markers",
            fillcolor=None,
            marker=dict(
                color="deepskyblue" if s.left_has_ball else "blue",
                size=2,
                opacity=0.3,
                symbol="circle-open",
            ),
            text="",
            hoverinfo="skip",
        )
    )

    # view of net intercepts
    for i, intercept in enumerate(active_intercepts):
        traces.append(
            go.Scatter(
                name=f"view_of_net_intercept_{i}",
                x=[s.active_pos[0], 1.0],
                y=[s.active_pos[1], intercept],
                mode="lines",
                line=dict(color="teal", width=1, dash="dot"),
                text="",
                hoverinfo="skip",
                visible=False,
            )
        )

    # main player markers
    traces.append(
        go.Scatter(
            name="left_team",
            x=s.team_pos[:, 0],
            y=s.team_pos[:, 1],
            mode="markers",
            marker=dict(color="blue", size=10),
            text=left_tooltip.get_text(),
            hoverinfo="text",
        )
    )
    traces.append(
        go.Scatter(
            name="right_team",
            x=s.opp_pos[:, 0],
            y=s.opp_pos[:, 1],
            mode="markers",
            marker=dict(color="red", size=10),
            text=right_tooltip.get_text(),
            hoverinfo="text",
        )
    )

    # player direction dots
    traces.append(
        go.Scatter(
            name="left_team_dir",
            x=s.team_pos[:, 0] + s.team_dir[:, 0],
            y=s.team_pos[:, 1] + s.team_dir[:, 1],
            mode="markers",
            marker=dict(color="deepskyblue", size=3, symbol="circle"),
            text="",
            hoverinfo="skip",
        )
    )
    traces.append(
        go.Scatter(
            name="right_team_dir",
            x=s.opp_pos[:, 0] + s.opp_dir[:, 0],
            y=s.opp_pos[:, 1] + s.opp_dir[:, 1],
            mode="markers",
            marker=dict(color="pink", size=3, symbol="circle"),
            text="",
            hoverinfo="skip",
        )
    )

    # line for active player
    traces.append(
        go.Scatter(
            name="left_line",
            x=[
                s.active_pos[0],
                s.active_pos[0] + 50 * s.active_dir[0],
            ],
            y=[
                s.active_pos[1],
                s.active_pos[1] + 50 * s.active_dir[1],
            ],
            mode="lines",
            line=dict(
                color="deepskyblue" if s.left_has_ball else "blue",
                width=0,
                dash="dot",
            ),
            text=right_tooltip.get_text(),
            hoverinfo="skip",
        )
    )

    # highlight ball carrier
    traces.append(
        go.Scatter(
            name="left_ball_carrier",
            x=[s.active_pos[0]],
            y=[s.active_pos[1]],
            mode="markers",
            marker=dict(
                color="deepskyblue",
                opacity=1 if s.left_has_ball else 0,
                size=10,
            ),
            text="",
            hoverinfo="skip",
        )
    )

    # the ball!
    traces.append(
        go.Scatter(
            name="ball",
            x=[s.ball_pos[0]],
            y=[s.ball_pos[1]],
            mode="markers",
            marker=dict(
                color="white",
                size=5 + 13 * s.ball_pos[2],
                opacity=min(max(0.15, 1 - s.ball_pos[2] / 9), 1.0),
                symbol="circle",
            ),
            text=ball_tooltip.get_text(),
            hoverinfo="text",
        )
    )

    # for trace in traces:
    #     trace.visible = True

    return traces


def make_figure_widget(n_traces=30):
    fig = go.FigureWidget()

    # pitch boundary
    fig.add_shape(
        type="rect",
        x0=-1.005,
        y0=0.42,
        x1=1.005,
        y1=-0.42,
        line=dict(color="white", width=1),
        layer="below",
    )

    # goals
    fig.add_shape(
        type="rect",
        x0=-1.02,
        y0=0.044,
        x1=-1.005,
        y1=-0.044,
        line=dict(color="silver", width=1),
        fillcolor="silver",
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=1.02,
        y0=0.044,
        x1=1.005,
        y1=-0.044,
        line=dict(color="silver", width=1),
        fillcolor="silver",
        layer="below",
    )

    fig.update_xaxes(
        range=[-1.05, 1.05],
        gridcolor="silver",
        zerolinecolor="silver",
        showgrid=False,
    )
    fig.update_yaxes(
        range=[0.45, -0.45],
        gridcolor="silver",
        zerolinecolor="silver",
        showgrid=False,
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="grey",
        autosize=False,
        width=900,
        height=400,
        margin=dict(l=10, r=10, b=10, t=40, pad=1),
    )

    for _ in range(n_traces):
        fig.add_scatter(x=[0], y=[0])  # , fillcolor="rgba(0,0,0,0)")

    return fig
