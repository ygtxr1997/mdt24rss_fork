import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter  # useful for `logit` scale


def exp_hd_ratio():
    # Provided real data
    data_ratio = [0., 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 1.]
    data_len = 201314 * 2  # assumed as a constant for all x_baseline values in your description
    x_baseline = np.array(data_ratio) * data_len
    y_baseline = np.array([1.310, 1.36, 1.527, 2.259, 2.993, 3.58, 3.7])

    # Data for "Ours"
    x_ours = np.array([0.])  # Tiny offset for display on log scale
    y_ours = np.array([1.75])

    # Plotting
    plt.figure(figsize=(8, 6))

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    # Plot Baseline data with line connecting points
    plt.scatter(x_baseline, y_baseline, color='b', label='Baseline', alpha=0.7, edgecolor='k', s=50)
    plt.plot(x_baseline, y_baseline, color='b', linestyle='-', linewidth=2)

    # Plot "Ours" data as a star marker
    plt.plot(np.concatenate([x_ours, x_baseline[-1:]]), np.concatenate([y_ours, y_ours]),
             color='r', linestyle='dashed', linewidth=2, alpha=0.7)
    plt.scatter(x_ours, y_ours, color='r', label='Ours', marker='*', s=400, edgecolor='k', alpha=0.7)
    plt.scatter(x_baseline[-1:], y_ours, color='r', marker='.', s=1, alpha=0.3)

    # Set x-axis to logarithmic scale, reversed from big to small
    plt.xscale('symlog')
    plt.gca().invert_xaxis()  # Reverse the x-axis

    # Styling for academic papers
    plt.xlabel('#Human Demonstrations in Target Domain', fontsize=20, weight='bold')
    plt.ylabel('Averaged Sequence Length', fontsize=20, weight='bold')
    # plt.title('Different ', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Major ticks font size
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks font size
    plt.tight_layout()

    # Save and show plot
    plt.savefig("plot/exp_hd_ratio.png", dpi=300)
    # plt.show()


def exp_img_ratio():
    # Provided real data
    data_ratio = [5e-4, 5e-3, 5e-2, 5e-1, 1.]
    data_len = 201314 * 2  # assumed as a constant for all x_baseline values in your description
    x_ours = np.array(data_ratio) * data_len
    y_ours = np.array([1.319, 1.456, 1.542, 1.649, 1.755])

    # Data for "Ours"
    x_baseline = np.array([0.])  # Tiny offset for display on log scale
    y_baseline = np.array([1.31])

    # Plotting
    plt.figure(figsize=(8, 6))

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    # Plot Baseline data with line connecting points
    plt.plot(x_ours, y_ours, color='r', linestyle='-', linewidth=2)
    plt.scatter(x_ours, y_ours, color='r', label='Ours', marker='*', alpha=0.7, edgecolor='k', s=200)

    # Plot "Ours" data as a star marker
    plt.plot(np.concatenate([x_baseline, x_ours[:1]]), np.concatenate([y_baseline, y_ours[:1]]),
             color='b', linestyle='dashed', linewidth=2, alpha=0.7)
    plt.scatter(x_baseline, y_baseline, color='b', label='Baseline', marker='.', s=200, edgecolor='k', alpha=0.7)
    # plt.scatter(x_ours[-1:], y_baseline, color='r', marker='.', s=1, alpha=0.3)

    # Set x-axis to logarithmic scale, reversed from big to small
    plt.xscale('symlog')
    plt.gca().invert_xaxis()  # Reverse the x-axis

    # Styling for academic papers
    plt.xlabel('#Images in Target Domain', fontsize=20, weight='bold')
    plt.ylabel('Averaged Sequence Length', fontsize=20, weight='bold')
    # plt.title('Different ', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Major ticks font size
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks font size
    plt.tight_layout()

    # Save and show plot
    plt.savefig("plot/exp_img_ratio.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    # exp_hd_ratio()
    exp_img_ratio()
