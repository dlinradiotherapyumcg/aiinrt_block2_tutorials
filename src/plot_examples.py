import numpy as np
import matplotlib.pyplot as plt


def plot_example_data(X_data, y_data, title="Example Data"):
    
    n_cols = np.unique(y_data).max()
    fig, axes = plt.subplots(4, n_cols + 1, figsize=(n_cols*2, 6))
    fig.suptitle(title, 
                fontsize=13, fontweight="bold", y=1.01)

    for digit in range(n_cols + 1):
        samples = X_data[y_data == digit][:4]

        if samples.ndim == 2:  # if the data has been flattened, unflatten it
            samples = samples.reshape(-1, 8, 8)
        for row, img in enumerate(samples):
            ax = axes[row, digit]
            ax.imshow(img, cmap="gray_r", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(str(digit), fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.show()