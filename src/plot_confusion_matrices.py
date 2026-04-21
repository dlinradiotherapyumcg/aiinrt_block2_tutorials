import matplotlib.pyplot as plt
from sklearn import metrics


def plot_two_confusion_matrices(y_true, y_pred_left, label_left, y_pred_right, label_right):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_left, ax=axes[0], colorbar=False
    )
    axes[0].set_title(label_left)

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_right, ax=axes[1], colorbar=False
    )
    axes[1].set_title(label_right)

    fig.suptitle("Confusion Matrices", y=1.02)
    plt.tight_layout()
    plt.show()