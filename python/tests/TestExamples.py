# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Run every script of examples/ for a few steps on the Reference platform, so that they keep working."""

import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples')

CASES = [
    ('particle/kick.py', ['--nodes', '12', '--steps', '5']),
    ('particle/kick.py', ['--preset', 'alanine', '--nodes', '12', '--steps', '5']),
    ('particle/kick.py', ['--nodes', '12', '--steps', '5', '--no-lb']),
    ('particle/thermal.py', ['--steps', '20', '--seed', '3']),
    ('particle/thermal.py', ['--beads', '4', '--steps', '20', '--seed', '3', '--removal', '1']),
    ('particle/uniform_flow.py', ['--nodes', '12', '--steps', '20', '--interval', '10']),
    ('fluid/initial_state.py', ['--nodes', '16', '--steps', '20', '--interval', '10', '--save', 'state.npz']),
    ('cocomo/diffusion.py', ['--preset', 'smoke', '--box', '10', '--steps', '20', '--report', '10']),
    ('cocomo/diffusion.py', ['--preset', 'smoke', '--steps', '20', '--report', '10', '--no-lb']),
    ('cocomo/diffusion.py', ['--preset', 'rlp', '--box', '10', '--steps', '20', '--report', '10']),
    ('cocomo/kick.py', ['--preset', 'peptide', '--nodes', '24', '--steps', '5']),
    ('cocomo/kick.py', ['--preset', 'ubiquitin', '--nodes', '24', '--steps', '5']),
    ('cocomo/kick.py', ['--preset', 'ubiquitin', '--nodes', '24', '--steps', '5', '--no-lb']),
]


@pytest.mark.parametrize('script,arguments', CASES, ids=['%s %s' % (s, ' '.join(a)) for s, a in CASES])
def test_example_runs(script, arguments, tmp_path):
    command = [sys.executable, os.path.join(EXAMPLES, script), '--platform', 'Reference'] + arguments
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fluid_state_restart(tmp_path):
    """Loading the saved state continues the run exactly: two runs of 20 steps give the run of 40 steps."""
    script = os.path.join(EXAMPLES, 'fluid', 'initial_state.py')
    common = [sys.executable, script, '--platform', 'Reference', '--nodes', '16', '--interval', '20']
    def run(arguments):
        result = subprocess.run(common + arguments, cwd=tmp_path, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.splitlines()
    whole = run(['--steps', '40'])
    run(['--steps', '20', '--save', 'half.npz'])
    second = run(['--steps', '20', '--load', 'half.npz'])
    assert whole[-1] == second[-1]


def test_plot(tmp_path):
    pytest.importorskip('matplotlib')
    data = tmp_path/'data.txt'
    data.write_text('# t v\n0 1\n1 0.5\n2 0.25\n')
    command = [sys.executable, os.path.join(EXAMPLES, 'plot.py'), str(data), '--logy', '--output', 'p.png']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path/'p.png').exists()


def test_msd(tmp_path):
    """msd.py recovers the diffusion coefficient of a random walk, within its statistical error."""
    import numpy as np
    rng = np.random.default_rng(0)
    D, dt = 0.002, 100.0
    x = np.cumsum(rng.normal(0, np.sqrt(2*D*dt), (20000, 3)), axis=0)
    np.savetxt(tmp_path/'com.txt', np.column_stack([dt*np.arange(len(x)), x]))
    command = [sys.executable, os.path.join(EXAMPLES, 'cocomo', 'msd.py'), 'com.txt']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith('D = ')][0]
    assert abs(float(line.split()[2]) - D*1e3) < 0.2*D*1e3


def test_cocomo2_parameters():
    """The COCOMO2 builder: masses, exclusions and elastic network of SOD1."""
    sys.path.insert(0, os.path.join(EXAMPLES, 'cocomo'))
    import numpy as np
    import cocomo2
    names, chains, positions = cocomo2.read_beads(os.path.join(EXAMPLES, 'cocomo', 'data', 'sod1.pdb'))
    assert len(names) == 110 and set(chains) == {'P001'}
    pairs = cocomo2.elastic_network(positions, [(2, 109)])
    assert len(pairs) == 478
    assert all(j - i >= 3 and r0 < 0.9 for i, j, r0 in pairs)
    xi = cocomo2.exposure(names, np.loadtxt(os.path.join(EXAMPLES, 'cocomo', 'data', 'sod1.surface')), [(1, 108)])
    assert xi[108] == xi[109] == 1 and 0 <= xi.min() and xi.max() <= 1
    system = cocomo2.create_system(names, chains, 15.0, xi, pairs)
    assert abs(sum(system.getParticleMass(i)._value for i in range(110)) - sum(cocomo2.RESIDUES[n][0] for n in names)) < 1e-9
