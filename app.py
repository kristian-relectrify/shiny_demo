import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_widget
from shiny.types import FileInfo

app_ui = ui.page_fluid(
    ui.panel_title("AC Stack Debug"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file(
                "file1",
                "Choose CSV File",
                accept=[".csv", ".log"],
                multiple=False,
                button_label="Upload",
                width=12,
            ),
            ui.input_selectize(
                "waveforms_to_plot",
                "Select waveforms to plot",
                choices=[],
                multiple=True,
            ),
            ui.input_checkbox("normalize", "Normalize", False),
            width=3,
        ),
        ui.panel_main(ui.output_ui("contents"), output_widget("ac_debug")),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect()
    def _():
        if input.file1() is not None:
            f: list[FileInfo] = input.file1()
            d_waveform = read_ac_logs_waveforms(f[0]["datapath"])
            ui.update_select(
                "waveforms_to_plot",
                label="Select waveforms to plot",
                choices=d_waveform.columns.to_list()[:-1],
                selected=None,
            )

    @output
    @render.ui
    def contents():
        if input.file1() is None:
            return "Please upload a csv file"
        f: list[FileInfo] = input.file1()
        d_waveform = read_ac_logs_waveforms(f[0]["datapath"])
        df_agg = d_waveform.agg(["max", "min"]).T
        return ui.HTML(df_agg.to_html(classes="table table-striped"))

    @output
    @render_widget
    def ac_debug():
        if input.file1() is None:
            return go.Figure()
        f: list[FileInfo] = input.file1()
        d_waveform = read_ac_logs_waveforms(f[0]["datapath"])
        d_state = read_ac_logs_state(f[0]["datapath"])
        return plot_ac_logs(
            d_waveform,
            d_state,
            waveforms_to_plot=input.waveforms_to_plot(),
            normalize=input.normalize(),
        )

    def read_ac_logs_waveforms(
        f_data: str,  # file path.
    ) -> pd.DataFrame:  # waveform dataframe
        """Get waveform data from file"""
        d = pd.read_csv(
            f_data,
            sep=",",
        )
        d.columns = [c.strip() for c in d.columns]

        # extract and process waveform data.
        d_waveform = d.drop("block_state", axis=1)
        d_waveform["search_time"] = d_waveform["search_time"]
        d_waveform["time"] = list(range(d_waveform.shape[0]))

        return d_waveform

    def read_ac_logs_state(
        f_data: str,  # file path.
    ) -> pd.DataFrame:  # state dataframe
        """Get state data from file"""
        d = pd.read_csv(
            f_data,
            sep=",",
        )
        d.columns = [c.strip() for c in d.columns]
        # extract and process block state data.
        d_state = state_data_to_line(d["block_state"])
        d_state["state"] = pd.Categorical(
            d_state["state"], categories=["A", "B", "C"], ordered=True
        )

        return d_state

    def parse_state(x) -> list:
        state_map = {
            "a": 0,
            "b": 1,
            "c": 2,
        }
        blocks_state = x.strip()[1:].split("-")

        switch_state = [s[1:-1].strip().split("|") for s in blocks_state]
        switch_state_ = []
        for i in range(len(switch_state)):
            switch_state_.append([int(state_map[s.lower()]) for s in switch_state[i]])

        return switch_state_

    def state_data_to_line(
        d: pd.DataFrame,  # raw state data
    ) -> pd.DataFrame:  # processed state data for point plots
        """Prepare state data for point plots"""

        # extract state data per block
        block_states = pd.DataFrame(d.apply(parse_state).to_list())
        n_blocks = block_states.shape[1]
        state_map_inv = {
            0: "A",
            1: "B",
            2: "C",
        }

        # get transition points for each block
        transition_points = []
        for i in range(n_blocks):
            temp_ = pd.DataFrame(block_states[i].to_list())
            # add final row to capture last transition.
            temp_ = pd.concat(
                [temp_, pd.DataFrame(np.ones((1, temp_.shape[1])) * 100)], axis=0
            ).reset_index(drop=True)
            transition_points.append(temp_.diff())

        # make a dataframe of all transition points with corresponding state
        transition_points_ = []
        for b_, block_transition_points in enumerate(transition_points):
            block_states_ = block_states[b_]

            for n_ in range(block_transition_points.shape[1]):
                node_transition_points = block_transition_points[n_]
                pts = node_transition_points[node_transition_points != 0]

                for p_ in range(len(pts) - 1):
                    transition_points_.append(
                        {
                            "block": b_,
                            "node": -1 * len(block_states_[0]) * b_ + -1 * n_,
                            "start": pts.index[p_],
                            "end": pts.index[p_ + 1] - 1,
                            "state": state_map_inv[block_states_[pts.index[p_]][n_]],
                        }
                    )

        return pd.DataFrame(transition_points_)

    # Function to plot using Plotly
    def plot_ac_logs(
        d_waveform: pd.DataFrame,
        d_state: pd.DataFrame,
        waveforms_to_plot: list,
        normalize: bool = False,
    ):
        """
        Plots waveform and stat data as plotly interactive plots.
        """
        n_blocks = d_state["block"].nunique()

        fig = sp.make_subplots(
            rows=1 + n_blocks,
            cols=1,
            row_heights=[2] + [1] * n_blocks,
            shared_xaxes=True,
        )

        # Plot waveform
        for wv in waveforms_to_plot:
            fig.add_trace(
                go.Scatter(
                    x=d_waveform["time"],
                    y=d_waveform[wv]
                    if not normalize
                    else (
                        (d_waveform[wv] - d_waveform[wv].min())
                        / (d_waveform[wv].max() - d_waveform[wv].min())
                    ),
                    mode="lines",
                    name=wv,
                )
            )

        fig.update_yaxes(gridcolor="rgba(0,0,0,0.2)")

        # Define a color scale for the legend
        color_scale = {"A": "red", "B": "green", "C": "blue"}

        # Plot node state
        for k, block in enumerate(d_state["block"].unique()):
            block_state_data = d_state[d_state["block"] == block]
            for index, row in block_state_data.iterrows():
                scatter = go.Scatter(
                    x=[row["start"], row["end"]],
                    y=[row["node"], row["node"]],
                    mode="lines+markers",
                    marker=dict(
                        size=6,
                        color=color_scale[row["state"]],
                    ),
                    showlegend=False,
                    legendgroup=row["state"],
                )
                fig.add_trace(scatter, row=k + 2, col=1)
                fig.update_yaxes(gridcolor="rgba(0,0,0,0.2)", row=k + 2, col=1)

        fig.update_layout(height=300 + 150 * n_blocks, showlegend=True)
        fig.update_xaxes(showticklabels=False)

        return fig


app = App(app_ui, server)
