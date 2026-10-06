# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Plot two columns of the text files written by the examples, and save the figure as a PNG image.

Columns are numbered from 1; lines that start with # are skipped.  For example, the speed of the bead
against time in the kick experiment, with and without the fluid, on a logarithmic scale:

    python plot.py kick_bead_lb_on.txt kick_bead_lb_off.txt --x 1 --y 5 --logy --output kick.png

Needs matplotlib (conda install -c conda-forge matplotlib).  The image is written to a file, so the
script works also on a remote computer without a display.
"""

import argparse

import matplotlib
matplotlib.use('Agg')                  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='text files with columns of numbers')
    parser.add_argument('--x', type=int, default=1, help='column on the horizontal axis (default 1)')
    parser.add_argument('--y', type=int, default=2, help='column on the vertical axis (default 2)')
    parser.add_argument('--logx', action='store_true', help='logarithmic horizontal axis')
    parser.add_argument('--logy', action='store_true', help='logarithmic vertical axis')
    parser.add_argument('--xlabel', help='label of the horizontal axis')
    parser.add_argument('--ylabel', help='label of the vertical axis')
    parser.add_argument('--output', default='plot.png', help='image file (default plot.png)')
    args = parser.parse_args()

    for name in args.files:
        data = np.loadtxt(name, ndmin=2)
        x, y = data[:, args.x - 1], data[:, args.y - 1]
        if args.logy:
            y = np.abs(y)
        plt.plot(x, y, label=name)
    if args.logx:
        plt.xscale('log')
    if args.logy:
        plt.yscale('log')
    plt.xlabel(args.xlabel or 'column %d' % args.x)
    plt.ylabel(args.ylabel or 'column %d' % args.y)
    plt.legend()
    plt.savefig(args.output, dpi=120)
    print('Figure written to', args.output)


if __name__ == '__main__':
    main()
