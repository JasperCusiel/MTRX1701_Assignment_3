import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator


def ellipse_func(x, y, x_0, y_0, a, b):
    # function: ellipse_func
    # inputs:
    #   x - x position of sensor
    #   y - y position of sensor
    #   x_0 - x position of centre of ellipse
    #   y_0 - y position of centre of ellipse
    #   a - major axis of ellipse
    #   b - minor axis of ellipse
    # return:
    #   F - F = ((x-x_0)^2)/a^2 + ((y-y_0)^2)/b^2 -1
    F = np.square((x - x_0) / a) + np.square((y - y_0) / b) - 1

    return F


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


if __name__ == '__main__':
    a = 12.5
    b = 7.5
    delta = 0.025

    # Create meshgrid
    X = np.arange(-20, 20, delta)
    Y = np.arange(-10, 25, delta)
    X, Y = np.meshgrid(X, Y)

    # Compute Z using the ellipse function
    Z = ellipse_func(X, Y, 0, b, a, b)

    # Create the plot
    fig, ax = plt.subplots()

    # Set plot parameters
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["font.family"] = "serif"
    ax.set_aspect(1)

    # Set plot limits
    plt.xlim(-20, 20)
    plt.ylim(-10, 25)

    # Set colour map to use
    cmap = plt.get_cmap('plasma')

    # Get max and min Z values
    z = Z[:-1, :-1]
    levels = MaxNLocator(nbins=80).tick_values(z.min(), z.max())
    # Draw contour map
    contour_map = ax.contourf(X, Y, Z,
                              norm=MidpointNormalize(midpoint=0.),
                              cmap=cmap, levels=levels)

    # Add in ellipse to draw track outline
    outer_ellipse = patches.Ellipse((0, b), 2 * a, 2 * b, edgecolor='white', fc='None', lw=2, label="Ellipse",
                                    linestyle=(10, (5, 2)))
    ax.add_patch(outer_ellipse)

    # Plot formatting
    plt.title(label="2D Plot Coloured By Distance To Ellipse", pad=14,
              fontdict={'family': 'serif', 'color': 'black', 'size': 14}, weight='bold')
    subtitle = "a = {}cm, b = {}cm, Centered at ({},{})".format(a, b, 0, b)
    plt.suptitle(t=subtitle, y=0.05, size='small')
    plt.xlabel("x (cm)", fontname='Times New Roman')
    plt.ylabel("y (cm)", fontname='Times New Roman')
    plt.legend(loc='upper right', fontsize=8)

    # Set the font to Times New Roman for axis numbers
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    # Make space for bottom text
    fig.subplots_adjust(bottom=0.17)

    # Add side colour bar
    cbar = fig.colorbar(contour_map, ax=ax, extend='both')
    cbar.set_label('Point Outside Of Ellipse Z > 0', rotation=90,
                   fontdict={'family': 'serif', 'color': 'black', 'size': 8})

    plt.savefig('question-2g-ellipse-plot.png')
