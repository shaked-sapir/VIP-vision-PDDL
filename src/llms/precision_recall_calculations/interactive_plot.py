import plotly.graph_objects as go


def plot_pr_interactive(
        precision: list[float], recall: list[float], thresholds: list[float], ap: float, title: str, save_path: str
):
    """
    Saves an interactive Plotly PR curve with threshold tooltips.

    Parameters:
        precision, recall, thresholds: Output from precision_recall_curve
        ap (float): Average precision
        title (str): Title
        save_path (str): Output HTML path
    """
    precision = precision[1:]  # skip first point (no threshold)
    recall = recall[1:]

    fig = go.Figure(data=go.Scatter(
        x=recall,
        y=precision,
        mode='lines+markers',
        marker=dict(size=6),
        text=[f"Threshold: {t:.2f}" for t in thresholds],
        hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=f"{title}<br>(AP = {ap:.2f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=600
    )

    fig.write_html(save_path)
