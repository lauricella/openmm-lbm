# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Mean square displacement and diffusion coefficient of the centre of mass.

Reads the file <prefix>_com.txt written by diffusion.py (time in ps, centre of mass x y z in nm, not
wrapped into the box), computes the mean square displacement MSD(t) averaged over all time origins,
for lag times up to a tenth of the run, and fits MSD = 6 D t + c over a range of lag times.  The
components are fitted as MSD_x = 2 D_x t + c.

    python msd.py sod1_lb_on_com.txt

The MSD is written to <input>.msd (lag time in ps, MSD total, x, y, z in nm^2).  For a free particle
of mass M with friction gamma and no hydrodynamics, D = kT/(M gamma).
"""

import argparse

import numpy as np


def msd(positions, maxLag):
    """MSD of the positions (frames x 3) for lags 0..maxLag frames, averaged over all time origins;
    returns an array (maxLag + 1) x 3 with the three components."""
    result = np.zeros((maxLag + 1, 3))
    for lag in range(1, maxLag + 1):
        d = positions[lag:] - positions[:-lag]
        result[lag] = (d**2).mean(axis=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('com', help='file with time (ps) and centre of mass x y z (nm)')
    parser.add_argument('--max-lag', type=float, default=0.1,
                        help='largest lag time, as a fraction of the run (default 0.1)')
    parser.add_argument('--fit', type=float, nargs=2, default=[0.1, 0.9], metavar=('FROM', 'TO'),
                        help='range of the fit, as fractions of the largest lag time (default 0.1 0.9)')
    args = parser.parse_args()

    data = np.loadtxt(args.com, ndmin=2)
    t, x = data[:, 0], data[:, 1:4]
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError('the times must be equally spaced')
    maxLag = int(args.max_lag*(len(t) - 1))
    if maxLag < 2:
        raise ValueError('the run is too short: %d frames' % len(t))
    components = msd(x, maxLag)
    lags = dt*np.arange(maxLag + 1)
    total = components.sum(axis=1)
    np.savetxt(args.com + '.msd', np.column_stack([lags, total, components]),
               header='lag time (ps), MSD total, x, y, z (nm^2)', fmt='%.6g')

    first, last = int(args.fit[0]*maxLag), int(args.fit[1]*maxLag)
    fitted = slice(max(first, 1), last + 1)
    slope = np.polyfit(lags[fitted], total[fitted], 1)[0]
    slopes = [np.polyfit(lags[fitted], components[fitted, k], 1)[0] for k in range(3)]
    print('%d frames every %g ps; lags up to %g ps; fit from %g to %g ps' % (
        len(t), dt, lags[-1], lags[fitted][0], lags[fitted][-1]))
    # 1 nm^2/ps = 1e3 nm^2/ns = 1e5 A^2/ns
    print('D = %.4g nm^2/ns = %.4g A^2/ns' % (slope/6*1e3, slope/6*1e5))
    print('D_x, D_y, D_z = %.4g, %.4g, %.4g nm^2/ns' % tuple(s/2*1e3 for s in slopes))
    print('MSD written to', args.com + '.msd')


if __name__ == '__main__':
    main()
