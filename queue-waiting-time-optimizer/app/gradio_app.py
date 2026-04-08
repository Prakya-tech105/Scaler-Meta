from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from qwt_optimizer.agents.rule_based import RuleBasedAgent
from qwt_optimizer.core import Action, SCENARIOS, DEFAULT_SEED
from qwt_optimizer.envs.queue_gym_env import QueueGymEnv
from qwt_optimizer.utils.seeding import set_global_seed


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.suffix == ".zip":
        return path
    if path.with_suffix(".zip").exists():
        return path.with_suffix(".zip")
    return path


def _load_dqn_model(model_path: str):
    try:
        from stable_baselines3 import DQN
    except Exception:
        return None

    resolved = _resolve_model_path(model_path)
    if not resolved.exists():
        return None

    return DQN.load(str(resolved))


def run_simulation(
    scenario_name: str,
    policy_name: str,
    seed: int,
    steps: int,
    model_path: str,
) -> tuple[str, go.Figure, pd.DataFrame]:
    set_global_seed(seed)

    env = QueueGymEnv(scenario_name=scenario_name, seed=seed)
    obs, info = env.reset(seed=seed)
    baseline = RuleBasedAgent()
    dqn_model = _load_dqn_model(model_path) if policy_name == "dqn" else None

    rows: list[dict[str, Any]] = []
    cumulative_reward = 0.0
    fallback_note = ""

    if policy_name == "dqn" and dqn_model is None:
        fallback_note = "DQN model not found; falling back to rule-based baseline for this run."

    for step_idx in range(steps):
        state_dict = info["state_dict"]

        if policy_name == "baseline" or dqn_model is None:
            action = int(baseline.act(state_dict))
            policy_used = "baseline"
        elif policy_name == "random":
            action = int(env.action_space.sample())
            policy_used = "random"
        else:
            action_pred, _ = dqn_model.predict(obs, deterministic=True)
            action = int(action_pred)
            policy_used = "dqn"

        obs, reward, terminated, truncated, step_info = env.step(action)
        cumulative_reward += float(reward)

        state = step_info["state_dict"]
        rows.append(
            {
                "step": step_idx + 1,
                "policy": policy_used,
                "action": int(action),
                "action_name": Action(int(action)).name,
                "queue_length": float(state["queue_length"]),
                "average_waiting_time": float(state["average_waiting_time"]),
                "open_counters": float(state["open_counters"]),
                "incoming_rate": float(state["incoming_rate"]),
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
            }
        )

        info = step_info
        if terminated or truncated:
            break

    env.close()

    df = pd.DataFrame(rows)
    if df.empty:
        summary = "No simulation steps were produced."
    else:
        final_row = df.iloc[-1]
        summary = (
            f"Scenario: {scenario_name} | Policy: {policy_name} | Seed: {seed} | Steps: {len(df)} | "
            f"Final queue: {final_row['queue_length']:.1f} | Final wait: {final_row['average_waiting_time']:.2f} | "
            f"Total reward: {df['reward'].sum():.2f}"
        )

    if fallback_note:
        summary = f"{summary}\n\n{fallback_note}"

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Queue Length",
            "Average Waiting Time",
            "Open Counters",
            "Cumulative Reward",
        ),
    )

    if not df.empty:
        fig.add_trace(go.Scatter(x=df["step"], y=df["queue_length"], name="Queue Length"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["step"], y=df["average_waiting_time"], name="Avg Waiting"), row=1, col=2)
        fig.add_trace(go.Scatter(x=df["step"], y=df["open_counters"], name="Open Counters"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["step"], y=df["cumulative_reward"], name="Cumulative Reward"), row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=30, r=30, t=50, b=30),
        template="plotly_white",
    )

    return summary, fig, df


def build_demo() -> gr.Blocks:
    css = """
    .qwt-shell {
        background: linear-gradient(180deg, #f7fafc 0%, #eef2f7 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 20px;
    }
    #table-container {
        min-height: 600px;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"), css=css) as demo:
        gr.Markdown(
            "# Queue Waiting Time Optimizer\n"
            "Interactive queue simulation for rule-based control and DQN policy comparison.")

        with gr.Row():
            with gr.Column(scale=1):
                scenario = gr.Dropdown(choices=list(SCENARIOS.keys()), value="medium", label="Scenario")
                policy = gr.Radio(choices=["baseline", "random", "dqn"], value="baseline", label="Policy")
                seed = gr.Slider(minimum=0, maximum=10_000, value=DEFAULT_SEED, step=1, label="Seed")
                steps = gr.Slider(minimum=5, maximum=200, value=50, step=1, label="Steps")
                model_path = gr.Textbox(
                    value="outputs/models/dqn_queue_optimizer_smoke.zip",
                    label="DQN model path",
                )
                run_btn = gr.Button("Run simulation", variant="primary")

            with gr.Column(scale=3):
                summary_box = gr.Markdown()
                plot_box = gr.Plot()

        with gr.Row():
            with gr.Column():
                gr.Markdown("## **Simulation trace**")
                table_box = gr.Dataframe(
                    headers=[
                        "step",
                        "policy",
                        "action",
                        "action_name",
                        "queue_length",
                        "average_waiting_time",
                        "open_counters",
                        "incoming_rate",
                        "reward",
                        "cumulative_reward",
                    ],
                    interactive=False,
                    elem_id="table-container",
                )

        run_btn.click(
            fn=run_simulation,
            inputs=[scenario, policy, seed, steps, model_path],
            outputs=[summary_box, plot_box, table_box],
        )

        gr.Markdown(
            "## Notes\n"
            "- Baseline uses queue thresholds.\n"
            "- DQN loads a saved model when available, otherwise the app falls back to the baseline.\n"
            "- Seeds are applied to Python, NumPy, and Torch when installed."
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
