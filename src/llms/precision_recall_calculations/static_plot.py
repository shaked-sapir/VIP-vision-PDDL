import matplotlib.pyplot as plt
import numpy as np


def plot_pr_static_with_thresholds(
        precision: list[float], recall: list[float], thresholds: list[float], ap: float, title: str, save_path: str,
        annotate_n=3):
    """
    Saves a static matplotlib PR curve with threshold annotations.

    Parameters:
        precision, recall, thresholds: Output from precision_recall_curve
        ap (float): Average precision
        title (str): Plot title
        save_path (str): Where to save image
        annotate_n (int): Number of threshold points to annotate
    """
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Annotate selected threshold points
    indices = np.linspace(0, len(thresholds) - 1, num=min(annotate_n, len(thresholds)), dtype=int)
    for i in indices:
        t = thresholds[i]
        p = precision[i + 1]
        r = recall[i + 1]
        plt.plot(r, p, 'ro')
        plt.text(r + 0.01, p + 0.01, f"{t:.2f}", fontsize=8)

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
