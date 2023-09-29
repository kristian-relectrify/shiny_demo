import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_widget
from shiny.types import FileInfo

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file(
                "file1", "Choose CSV File", accept=[".csv", ".log"], multiple=False
            ),
            ui.input_select(
                "waveforms_to_plot", "Select waveforms to plot", [], multiple=True
            ),
            ui.input_checkbox("normalize", "Normalize", False),
            ui.input_checkbox("hide_state", "Hide State", False),
            width=5,
        ),
        ui.panel_main(ui.output_ui("contents"), output_widget("ac_debug")),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect()
    def _():
        if input.file1() is not None:
            f: list[FileInfo] = input.file1()
            a, b = read_ac_logs(f[0]["datapath"])
            ui.update_select(
                "waveforms_to_plot",
                label="Select waveforms to plot",
                choices=a.columns.to_list()[:-1],
                selected=None,
            )

    @output
    @render.ui
    def contents():
        if input.file1() is None:
            return "Please upload a csv file"
        f: list[FileInfo] = input.file1()
        a, b = read_ac_logs(f[0]["datapath"])
        df_agg = a.agg(["max", "min"]).T
        return ui.HTML(df_agg.to_html(classes="table table-striped"))

    @output
    @render_widget
    def ac_debug():
        if input.file1() is None:
            return go.Figure()
        f: list[FileInfo] = input.file1()
        a, b = read_ac_logs(f[0]["datapath"])
        return plot_ac_logs(
            a,
            b,
            waveforms_to_plot=input.waveforms_to_plot(),
            normalize=input.normalize(),
            hide_state=input.hide_state(),
        )

    def read_ac_logs(f_data: str):
        """
        Reads and process waveform and state data from logs
        """
        d = pd.read_csv(
            f_data,
            sep=",",
        )
        d.columns = [c.strip() for c in d.columns]

        # extract and process waveform data.
        d_waveform = d.drop("block_state", axis=1)
        d_waveform["search_time"] = d_waveform["search_time"]
        d_waveform["time"] = list(range(d_waveform.shape[0]))

        # extract and process block state data.
        d_state = process_state_data(d["block_state"])
        d_state["state"] = pd.Categorical(
            d_state["state"], categories=["A", "B", "C"], ordered=True
        )

        return d_waveform, d_state

    def process_state_data(d: pd.DataFrame, block_size: int = 8):
        """
        Process ac stack beta log files.

        """

        def get_state(x):
            state_map = {
                "a": 0,
                "b": 1,
                "c": 2,
            }
            blocks_state = x.strip()[1:].split("-")

            switch_state = [s[1:-1].strip().split("|") for s in blocks_state]
            switch_state_ = []
            for i in range(len(switch_state)):
                switch_state_.append(
                    [int(state_map[s.lower()]) for s in switch_state[i]]
                )

            return switch_state

        # reformats data to tidy data.
        new_df = list()
        for i, value in enumerate(d.apply(get_state)):
            for r in range(len(value)):
                for i_, c in enumerate(value[r]):
                    new_df.append(
                        {
                            "time": i,
                            "block": r,
                            "index": -block_size * r + -1 * i_,
                            "state": c,
                        }
                    )

        return pd.DataFrame(new_df)

    # Function to plot using Plotly
    def plot_ac_logs(
        d_waveform: pd.DataFrame,
        d_state: pd.DataFrame,
        waveforms_to_plot: list,
        normalize: bool = False,
        hide_state: bool = False,
    ):
        """
        Plots waveform and stat data as plotly interactive plots.
        """
        n_blocks = d_state["block"].nunique() if not hide_state else 0

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

        if not hide_state:
            # Define a color scale for the legend
            color_scale = {"A": "red", "B": "green", "C": "blue"}

            # Plot node state
            for k, block in enumerate(d_state["block"].unique()):
                state_data = d_state[d_state["block"] == block]

                for s_ in color_scale.keys():
                    state_data_ = state_data[state_data["state"] == s_]
                    scatter = go.Scatter(
                        x=state_data_["time"],
                        y=state_data_["index"],
                        legendgroup=s_,
                        showlegend=k == 2,
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=color_scale[s_],
                        ),
                        name=s_,  # Unique legend name for each color/state
                    )
                    fig.add_trace(scatter, row=k + 2, col=1)
                    fig.update_yaxes(gridcolor="rgba(0,0,0,0.2)", row=k + 2, col=1)

            fig.update_layout(height=300 + 150 * n_blocks, showlegend=True)
            fig.update_xaxes(showticklabels=False)

        return fig


app = App(app_ui, server)
