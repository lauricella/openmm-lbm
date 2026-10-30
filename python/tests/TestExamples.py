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
    ('particle/kick.py', ['--nodes', '12', '--steps', '5', '--drag', 'Centered']),
    ('particle/thermal.py', ['--beads', '4', '--steps', '20', '--seed', '3', '--drag', 'Centered']),
    ('particle/uniform_flow.py', ['--nodes', '12', '--steps', '20', '--interval', '10', '--drag', 'Centered']),
    ('cocomo/diffusion.py', ['--preset', 'smoke', '--box', '10', '--steps', '20', '--report', '10', '--drag', 'Centered']),
    ('cocomo/diffusion.py', ['--preset', 'smoke', '--box', '10', '--steps', '20', '--report', '10', '--drag', 'Centered',
                             '--fluid-fluctuations', '--vtk', '10']),
    ('cocomo/kick.py', ['--preset', 'peptide', '--nodes', '24', '--steps', '5', '--drag', 'Centered']),
    ('cocomo/kick.py', ['--preset', 'sod1', '--box', '9', '--steps', '5']),
    ('cocomo/kick.py', ['--preset', 'sod1', '--box', '9', '--steps', '5', '--domain', '1', '110']),
    ('cocomo/kick.py', ['--preset', 'sod1', '--box', '9', '--steps', '5', '--temperature', '300', '--fluid-fluctuations',
                        '--drag', 'Centered', '--equilibration', '5', '--repeats', '2']),
    # the viscosity and the temperature, which every simulation script takes (examples/README.md)
    ('particle/kick.py', ['--nodes', '12', '--steps', '5', '--viscosity', '3', '--temperature', '300']),
    ('particle/kick.py', ['--nodes', '12', '--steps', '5', '--temperature', '300', '--no-lb']),
    ('particle/thermal.py', ['--steps', '20', '--seed', '3', '--viscosity', '5', '--temperature', '0']),
    ('particle/uniform_flow.py', ['--nodes', '12', '--steps', '20', '--interval', '10', '--viscosity', '3',
                                  '--temperature', '300']),
    ('fluid/initial_state.py', ['--nodes', '16', '--steps', '20', '--interval', '10', '--temperature', '298']),
    ('cocomo/kick.py', ['--preset', 'peptide', '--nodes', '24', '--steps', '5', '--viscosity', '1.0035',
                        '--temperature', '298']),
    ('cocomo/kick.py', ['--preset', 'peptide', '--nodes', '24', '--steps', '5', '--temperature', '298', '--no-lb']),
    ('cocomo/diffusion.py', ['--preset', 'smoke', '--box', '10', '--steps', '20', '--report', '10', '--temperature',
                             '0']),
    ('cocomo/shear.py', ['--box', '10', '10', '--height', '12', '--steps', '20', '--report', '10']),
    ('cocomo/shear.py', ['--box', '10', '10', '--height', '12', '--steps', '20', '--report', '10', '--velocity', '1',
                         '--no-fluid-fluctuations', '--drag', 'Explicit']),
]


@pytest.mark.parametrize('script,arguments', CASES, ids=['%s %s' % (s, ' '.join(a)) for s, a in CASES])
def test_example_runs(script, arguments, tmp_path):
    command = [sys.executable, os.path.join(EXAMPLES, script), '--platform', 'Reference'] + arguments
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('script, first, second', [
    ('particle/thermal.py', ['--beads', '4'], ['--particles', '4']),
    ('particle/kick.py', ['--preset', 'bead'], ['--preset', 'particle'])], ids=['--particles', 'preset particle'])
def test_bead_and_particle_names(tmp_path, script, first, second):
    """The second names of the options (--particles, the preset particle) give the same run as the first ones."""
    common = [sys.executable, os.path.join(EXAMPLES, script), '--platform', 'Reference', '--nodes', '12',
              '--steps', '5']
    if script.endswith('thermal.py'):
        common += ['--seed', '3']
    outputs = []
    for i, names in enumerate([first, second]):
        result = subprocess.run(common + names + ['--output', 'run%d.txt' % i], cwd=tmp_path, capture_output=True,
                                text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append((tmp_path/('run%d.txt' % i)).read_text())
    assert outputs[0] == outputs[1]


def test_kick_repeats(tmp_path):
    """At zero temperature the kicks of --repeats are all the same: their mean is the single kick, with zero error."""
    import numpy as np
    common = [sys.executable, os.path.join(EXAMPLES, 'cocomo', 'kick.py'), '--platform', 'Reference', '--preset',
              'peptide', '--nodes', '24', '--steps', '10']
    for name, extra in (('one.txt', []), ('three.txt', ['--repeats', '3'])):
        result = subprocess.run(common + extra + ['--output', name], cwd=tmp_path, capture_output=True, text=True,
                                timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    one, three = np.loadtxt(tmp_path/'one.txt'), np.loadtxt(tmp_path/'three.txt')
    assert three.shape == (11, 10)
    assert np.array_equal(three[:, :9], one)
    assert np.all(three[:, 9] < 1e-12)         # zero up to the rounding of the mean of equal numbers


def _pdb(atoms):
    """PDB lines for (residue name, residue number, atom name, element, x, y, z in Angstrom)."""
    return ''.join('ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n' % (
        i + 1, name, res, num, x, y, z, element) for i, (res, num, name, element, x, y, z) in enumerate(atoms))


def test_surface_exact_cases():
    """surface.py against exact areas: an isolated atom, two overlapping spheres (spherical cap), a buried atom."""
    import math
    import numpy as np
    sys.path.insert(0, os.path.join(EXAMPLES, 'cocomo'))
    import surface
    R = surface.RADII['C'] + 0.14
    # one atom: the whole grown sphere
    areas = surface.atom_areas(np.zeros((1, 3)), np.array([surface.RADII['C']]))
    assert abs(areas[0] - 4*math.pi*R**2) < 1e-12
    # two atoms at a distance d < 2R: each loses a cap of height h = R - d/2; 50 random distances and orientations
    rng = np.random.default_rng(5)
    errors = []
    for _ in range(50):
        d, u = rng.uniform(0.05, 0.6), rng.normal(size=3)
        areas = surface.atom_areas(np.array([[0, 0, 0], d*u/np.linalg.norm(u)]), np.full(2, surface.RADII['C']))
        errors.append(np.abs(areas - (4*math.pi*R**2 - 2*math.pi*R*(R - d/2))).max()/(4*math.pi*R**2))
    assert np.mean(errors) < 3e-3 and np.max(errors) < 1e-2
    # an atom inside an octahedron of atoms 0.25 nm away is not accessible
    centre = [[0, 0, 0]] + [[s*0.25*(k == 0), s*0.25*(k == 1), s*0.25*(k == 2)] for k in range(3) for s in (1, -1)]
    areas = surface.atom_areas(np.array(centre, dtype=float), np.full(7, surface.RADII['C']))
    assert areas[0] == 0


def test_surface_script(tmp_path):
    """surface.py writes one area per residue, leaves out hydrogens and water, and refuses a coarse-grained file."""
    import numpy as np
    pdb = tmp_path/'two.pdb'
    pdb.write_text(_pdb([('ALA', 1, ' CA ', 'C', 0, 0, 0), ('ALA', 1, ' HA ', 'H', 1.0, 0, 0),
                         ('GLY', 2, ' CA ', 'C', 30, 0, 0), ('HOH', 3, ' O  ', 'O', 0, 30, 0)]))
    script = os.path.join(EXAMPLES, 'cocomo', 'surface.py')
    result = subprocess.run([sys.executable, script, str(pdb), '--output', 'two.surface'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    areas = np.loadtxt(tmp_path/'two.surface')
    assert areas.shape == (2,)
    assert np.allclose(areas, 4*np.pi*(0.17 + 0.14)**2, rtol=0, atol=1e-3)       # isolated carbons, no hydrogen
    result = subprocess.run([sys.executable, script, os.path.join(EXAMPLES, 'cocomo', 'data', 'sod1.pdb')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0 and 'all-atom' in result.stderr
    # --chain keeps one copy of a crystal structure: the touching atom of chain B no longer hides part of chain A
    pair = tmp_path/'pair.pdb'
    lines = _pdb([('ALA', 1, ' CA ', 'C', 0, 0, 0), ('ALA', 1, ' CA ', 'C', 4.0, 0, 0)]).splitlines(keepends=True)
    pair.write_text(lines[0] + lines[1][:21] + 'B' + lines[1][22:])          # second atom in chain B
    for chains, expected in (([], 2), (['--chain', 'A'], 1)):
        result = subprocess.run([sys.executable, script, str(pair), '--output', 'pair.surface'] + chains, cwd=tmp_path,
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        areas = np.atleast_1d(np.loadtxt(tmp_path/'pair.surface'))
        assert len(areas) == expected
    assert abs(areas[0] - 4*np.pi*(0.17 + 0.14)**2) < 1e-3         # the file has three decimals


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


@pytest.mark.parametrize('options, ranges', [([], [(2, 109)]),
                                              (['--domain', '1', '110'], [(1, 110)]),
                                              (['--domain', '1', '110', '--enm-domain', '2', '109'], [(2, 109)])],
                         ids=['preset', 'domain alone', 'both'])
def test_diffusion_domain(tmp_path, options, ranges):
    """--domain alone sets the residues of the elastic network too; the SOD1 presets and --enm-domain keep 2-109."""
    sys.path.insert(0, os.path.join(EXAMPLES, 'cocomo'))
    import re
    import cocomo2
    names, chains, positions = cocomo2.read_beads(os.path.join(EXAMPLES, 'cocomo', 'data', 'sod1.pdb'))
    expected = len(cocomo2.elastic_network(positions, ranges))
    script = os.path.join(EXAMPLES, 'cocomo', 'diffusion.py')
    command = [sys.executable, script, '--platform', 'Reference', '--preset', 'smoke', '--box', '10', '--steps', '2',
               '--report', '1', '--checkpoint', '0'] + options
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert int(re.search(r'(\d+) elastic bonds', result.stdout).group(1)) == expected


@pytest.mark.parametrize('fluctuations', [False, True], ids=['still fluid', 'fluctuating fluid'])
def test_diffusion_restart(tmp_path, fluctuations):
    """A run split in two with --checkpoint and --restart gives the same output as an uninterrupted run."""
    script = os.path.join(EXAMPLES, 'cocomo', 'diffusion.py')
    common = [sys.executable, script, '--platform', 'Reference', '--preset', 'smoke', '--box', '10', '--checkpoint', '20']
    suffixes = ['_com.txt', '_temperature.txt']
    if fluctuations:                                     # the fluid temperature every 10 reports: at 20, 40, 60
        common += ['--report', '2', '--drag', 'Centered', '--fluid-fluctuations']
        suffixes.append('_fluid_temperature.txt')
    else:
        common += ['--report', '10']
    def run(arguments):
        result = subprocess.run(common + arguments, cwd=tmp_path, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    run(['--steps', '60', '--output', 'whole'])
    run(['--steps', '30', '--output', 'split'])          # stops 10 steps after the checkpoint at step 20
    run(['--steps', '60', '--output', 'split', '--restart'])
    for suffix in suffixes:
        assert (tmp_path/('whole' + suffix)).read_text() == (tmp_path/('split' + suffix)).read_text()
    assert (tmp_path/'split_20.dcd').exists()


def test_diffusion_fluid_temperature(tmp_path):
    """With --fluid-fluctuations the fluid is at the temperature of the run: the mean of 10 values on 20^3 nodes
    is within 2% of 298 K (the statistical error is about 1 K)."""
    import numpy as np
    script = os.path.join(EXAMPLES, 'cocomo', 'diffusion.py')
    command = [sys.executable, script, '--platform', 'Reference', '--preset', 'smoke', '--box', '10', '--steps', '200',
               '--report', '2', '--drag', 'Centered', '--fluid-fluctuations', '--output', 'fluct']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    fluid = np.loadtxt(tmp_path/'fluct_fluid_temperature.txt', ndmin=2)
    assert len(fluid) == 10
    assert abs(fluid[:, 1].mean() - 298) < 0.02*298


def test_shear_profile(tmp_path):
    """shear.py starts the fluid with the steady Couette profile of its faces: at zero temperature the planes next to
    the faces, far from the protein, keep the linear profile u_x = v (2 (k + 1)/(n_z + 1) - 1) (the file has five
    decimals)."""
    import numpy as np
    command = [sys.executable, os.path.join(EXAMPLES, 'cocomo', 'shear.py'), '--platform', 'Reference', '--box', '10',
               '10', '--height', '12', '--temperature', '0', '--steps', '20', '--report', '10', '--profile', '1']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    profile = np.loadtxt(tmp_path/'shear_profile.txt', ndmin=2)
    nz, v = 24, 0.5
    linear = v*(2*(np.arange(nz) + 1)/(nz + 1) - 1)
    assert profile.shape == (2, nz + 1)
    for k in (0, 1, 2, nz - 3, nz - 2, nz - 1):
        assert np.all(np.abs(profile[:, 1 + k] - linear[k]) < 1e-5)


@pytest.mark.parametrize('options, message', [(['--velocity', '3'], 'Mach'),
                                               (['--report', '10', '--checkpoint', '25'], 'multiple')],
                         ids=['Mach', 'checkpoint'])
def test_shear_refuses(tmp_path, options, message):
    """shear.py refuses faces faster than Mach 0.1 (2.9 nm/ps with the default lattice) and checkpoints between
    two reports."""
    command = [sys.executable, os.path.join(EXAMPLES, 'cocomo', 'shear.py'), '--platform', 'Reference'] + options
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0 and message in result.stderr


def test_shear_restart(tmp_path):
    """A run of shear.py split in two with --checkpoint and --restart gives the same output as an uninterrupted run,
    the accumulated angle of rotation and the VTK files included: the collection lists every step once, and the files
    written after the checkpoint by the first part are written again, the same."""
    script = os.path.join(EXAMPLES, 'cocomo', 'shear.py')
    common = [sys.executable, script, '--platform', 'Reference', '--box', '10', '10', '--height', '12', '--report', '10',
              '--profile', '1', '--checkpoint', '20', '--vtk', '10']
    def run(arguments):
        result = subprocess.run(common + arguments, cwd=tmp_path, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    run(['--steps', '60', '--output', 'whole'])
    run(['--steps', '30', '--output', 'split'])          # stops 10 steps after the checkpoint at step 20
    run(['--steps', '60', '--output', 'split', '--restart'])
    for suffix in ('_motion.txt', '_profile.txt', '_temperature.txt'):
        assert (tmp_path/('whole' + suffix)).read_text() == (tmp_path/('split' + suffix)).read_text()
    assert (tmp_path/'split_20.dcd').exists()
    assert (tmp_path/'split.pvd').read_text().replace('split_', 'whole_') == (tmp_path/'whole.pvd').read_text()
    for step in range(10, 70, 10):
        for name in ('density_%010d.vti', 'velocity_%010d.vti', 'particles_%010d.vtp'):
            assert (tmp_path/('whole_' + name % step)).read_bytes() == (tmp_path/('split_' + name % step)).read_bytes()
