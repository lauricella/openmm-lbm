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
