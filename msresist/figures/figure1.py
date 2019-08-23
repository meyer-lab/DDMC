"""
This creates Figure 1.
"""
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 9), (4, 3))

    # blank out first axis for cartoon
    ax[0].axis('off')

    Y_cv1 = pd.read_csv('../data/Phenotypic_data/CV_raw3.csv').iloc[:30, :11]
	Y_cv2 = pd.read_csv('../data/Phenotypic_data/CV_raw4.csv').iloc[:29, :11]

    plotTimeCourse(ax[1:3], Y_cv1, Y_cv2)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotTimeCourse(axs):
	""" Plots the Incucyte timecourse. """
	axs[0].set_title("Experiment 3")
	axs[0].plot(Y_cv1.iloc[:, 0], Y_cv1.iloc[:, 1:])
	axs[0].set_xticks(Y_cv1.iloc[:, 0])
	axs[0].set_yticks(np.arange(0, 100, 10))
	axs[0].legend(Y_cv1.columns[1:])
	axs[0].set_ylabel("% Confluency")
	axs[0].set_xlabel("Time (hours)")
	axs[1].set_title("Experiment 4")
	axs[1].plot(Y_cv2.iloc[:, 0], Y_cv2.iloc[:, 1:])
	axs[1].set_xticks(Y_cv2.iloc[:, 0])
	axs[1].set_yticks(np.arange(0, 100, 10))
	axs[1].legend(Y_cv2.columns[1:])
	axs[1].set_ylabel("% Confluency")
	axs[1].set_xlabel("Time (hours)");

