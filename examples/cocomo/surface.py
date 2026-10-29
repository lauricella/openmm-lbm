# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Solvent accessible surface of each residue of a protein, the input of cocomo2.exposure().

COCOMO2 scales the nonbonded terms of a folded residue by its exposure to the solvent, S/S_ref, where S is the
solvent accessible surface area (SASA) of the residue in the all-atom structure of the protein.  This script
computes S from an all-atom PDB file and writes one value per residue, in nm^2, in the order of the residues in
the file: the format read by diffusion.py and kick.py (--surface).

Method: Shrake and Rupley, J. Mol. Biol. 79, 351 (1973).  Every atom is a sphere of its van der Waals radius
(Bondi, J. Phys. Chem. 68, 441 (1964); the table RADII below) grown by the radius of the probe, 0.14 nm for water.
The surface of each sphere is sampled with --points points spread evenly over it (a Fibonacci spiral); a point is
accessible when it lies outside every other grown sphere, and the accessible area of the atom is the fraction of
accessible points times the area of its sphere.  The area of a residue is the sum over its atoms.  With the default
960 points the error on the area of an atom, measured on two overlapping spheres at random distances and
orientations, is 0.14 % of the area of its sphere on average and at most 0.7 % (0.85 % and 3 % with 100 points).

The structure must be all-atom: the element of every atom is read from the element columns of the PDB file
(77-78), or else from the first letter of the atom name, and an atom whose element is not in RADII stops the
script.  A coarse-grained file, with one bead per residue, gives an error instead of a meaningless surface.
Hydrogen atoms are left out unless --hydrogens is given, so that structures with and without them give the same
surfaces; water and other residues that are not amino acids are left out; of alternate locations the first one
is kept.

Usage:
    python surface.py protein.pdb > protein.surface
    python surface.py protein.pdb --output protein.surface --list     # also prints residue and area
"""

import argparse
import math
import sys

import numpy as np

# Van der Waals radii in nm (Bondi 1964).
RADII = {'H': 0.120, 'C': 0.170, 'N': 0.155, 'O': 0.152, 'S': 0.180, 'SE': 0.190, 'P': 0.180}

AMINO_ACIDS = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
               'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
               'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX', 'ASH', 'GLH', 'LYN', 'MSE', 'SEC'}


def read_atoms(path, hydrogens=False):
    """Atoms of the amino acids in an all-atom PDB file: (positions in nm, radii in nm, residue index of each atom,
    list of residues as (chain, number, insertion code, name))."""
    positions, radii, owner, residues, index = [], [], [], [], {}
    with open(path) as lines:
        for line in lines:
            if line.startswith('ENDMDL'):          # first model only
                break
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            name, altloc, resname = line[12:16], line[16], line[17:20].strip()
            if resname not in AMINO_ACIDS or altloc not in (' ', 'A', '1'):
                continue
            element = line[76:78].strip().upper()
            if not element:
                element = 'SE' if name.strip() == 'SE' else name.strip().lstrip('0123456789')[:1].upper()
            if element in ('H', 'D') and not hydrogens:
                continue
            if element == 'D':
                element = 'H'
            if element not in RADII:
                raise ValueError('atom %r of residue %s %s: element %r not in the table of radii; the structure must '
                                 'be all-atom' % (name.strip(), resname, line[22:27].strip(), element))
            key = (line[21], int(line[22:26]), line[26], resname)
            if key not in index:
                index[key] = len(residues)
                residues.append(key)
            positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            radii.append(RADII[element])
            owner.append(index[key])
    if not positions:
        raise ValueError('no atoms of amino acids in %s' % path)
    return np.array(positions)/10.0, np.array(radii), np.array(owner), residues


def sphere_points(n):
    """n unit vectors spread evenly over the sphere (Fibonacci spiral)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0*k/n
    phi = math.pi*(3.0 - math.sqrt(5.0))*k
    r = np.sqrt(1.0 - z*z)
    return np.stack([r*np.cos(phi), r*np.sin(phi), z], axis=1)


def atom_areas(positions, radii, probe=0.14, points=960):
    """Solvent accessible area of every atom (nm^2), by the Shrake-Rupley method."""
    grown = radii + probe
    unit = sphere_points(points)
    # Neighbour search on a grid of cells as large as the largest possible distance between overlapping spheres.
    size = 2.0*grown.max()
    cells = {}
    keys = np.floor(positions/size).astype(int)
    for i, key in enumerate(map(tuple, keys)):
        cells.setdefault(key, []).append(i)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    areas = np.empty(len(positions))
    for i in range(len(positions)):
        candidates = [j for o in offsets for j in cells.get((keys[i][0] + o[0], keys[i][1] + o[1], keys[i][2] + o[2]),
                                                            ()) if j != i]
        candidates = np.array(candidates, dtype=int)
        if len(candidates):
            d = np.linalg.norm(positions[candidates] - positions[i], axis=1)
            candidates = candidates[d < grown[candidates] + grown[i]]
        surface = positions[i] + grown[i]*unit
        free = np.ones(points, dtype=bool)
        for j in candidates:
            free &= ((surface - positions[j])**2).sum(axis=1) >= grown[j]**2
        areas[i] = 4.0*math.pi*grown[i]**2*free.mean()
    return areas


def residue_areas(path, probe=0.14, points=960, hydrogens=False):
    """Solvent accessible area of every residue (nm^2) and the list of residues."""
    positions, radii, owner, residues = read_atoms(path, hydrogens)
    areas = atom_areas(positions, radii, probe, points)
    return np.bincount(owner, weights=areas, minlength=len(residues)), residues


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pdb', help='all-atom structure of the protein')
    parser.add_argument('--output', help='file of the surfaces, one value per residue in nm^2 (default: standard output)')
    parser.add_argument('--probe', type=float, default=0.14, help='radius of the probe in nm (default 0.14, water)')
    parser.add_argument('--points', type=int, default=960, help='points on the sphere of each atom (default 960)')
    parser.add_argument('--hydrogens', action='store_true', help='include the hydrogen atoms (default: left out)')
    parser.add_argument('--list', action='store_true', help='print residue, number and area on the standard error')
    args = parser.parse_args()
    areas, residues = residue_areas(args.pdb, args.probe, args.points, args.hydrogens)
    out = open(args.output, 'w') if args.output else sys.stdout
    for area in areas:
        print('%.3f' % area, file=out)
    if args.output:
        out.close()
    if args.list:
        for (chain, number, code, name), area in zip(residues, areas):
            print('%s %4d%s %s %.3f' % (chain, number, code.strip(), name, area), file=sys.stderr)
    print('%d residues, total surface %.2f nm^2' % (len(residues), areas.sum()), file=sys.stderr)


if __name__ == '__main__':
    main()
