# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""The COCOMO2 coarse-grained model of proteins for OpenMM: one bead per residue.

Written for these examples from the description of the model in

    A. Jussupow, D. Bartley, L. J. Lapidus, M. Feig, "COCOMO2: A coarse-grained model for interacting
    folded and disordered proteins", J. Chem. Theory Comput. 21, 2095-2107 (2025) (CC-BY 4.0),

which extends COCOMO (G. Valdes-Garcia, L. Heo, L. J. Lapidus, M. Feig, J. Chem. Theory Comput. 19,
669-678 (2023)).  Only proteins are covered, not nucleic acids.  The energy is

    U = U_bond + U_angle + U_ENM + U_short-range + U_electrostatic

- bonds between consecutive beads: harmonic, l0 = 0.38 nm, k = 4184 kJ/mol/nm^2;
- angles of three consecutive beads: harmonic, theta0 = 180 degrees, k = 4.184 kJ/mol/rad^2;
- elastic network inside folded domains: harmonic, k = 500 kJ/mol/nm^2, r0 the distance in the
  reference structure, for the pairs closer than 0.9 nm and at least three residues apart;
- short range: xi_ij 4 eps_ij [(sigma_ij/r)^10 - (sigma_ij/r)^5], sigma_i = 2 r_i 2^(-1/6),
  sigma_ij the arithmetic mean, eps_ij the geometric mean of eps_polar = 0.176 kJ/mol and
  eps_hydrophobic = 0.295 kJ/mol; an extra term of the same form with eps = 0.3 kJ/mol between
  Arg/Lys and Phe/Tyr/Trp (cation-pi) and with eps = 0.1 kJ/mol between aromatic residues (pi-pi);
- electrostatics: xi_ij (A_i A_j + A0_i + A0_j) exp(-r/kappa)/r, A_i = sign(q_i) sqrt(0.75 |q_i|),
  A0 = 0 for polar and 0.0002 kJ nm/mol for hydrophobic residues, kappa = 1 nm;
- all nonbonded terms cut at 3 nm with periodic boundaries, without switching, and excluded between
  bonded beads.

xi_ij = sqrt(xi_i xi_j) scales the nonbonded terms by the exposure of the residues: xi_i = 1 in
disordered regions, and min(S_i/S_ref,i, lambda)/lambda in folded domains, with S_i the solvent
accessible surface of residue i in the reference structure, S_ref,i that of the same amino acid in
an alanine helix, and lambda = 0.7 (eq 6 of the article).

Differences from the text of the article:
- the cation-pi and pi-pi terms, written in eq 5 as eps_ij + eps_mod inside the 10-5 potential, are a
  separate potential of the same form with eps_mod, which is the same energy;
- A0_ij is the sum A0_i + A0_j, while eq 7 writes the product.  With A0 = 0 or 0.0002 kJ nm/mol the
  difference is at most 4e-4 kJ nm/mol;
- eq 7 has a factor rho in front of the electrostatic term that the article does not define; it is 1
  here.

The table below gives, for each amino acid, the mass (Da), the charge (e), the radius r of the sphere
of the same volume (nm), and S_ref (nm^2, Table S2 of the supporting information of the article).

Several proteins.  A structure may hold several chains (several proteins, or several copies of one): read_beads()
gives the chain of every bead, bonds and angles join only consecutive beads of the same chain, and
chain_ranges() lists the chains in the order of the file.  Every folded chain has its own folded domains, given
by read_domains() from a file with one line per chain, in residue numbers counted from 1 inside each chain and
converted to the numbering of the whole structure; exposure() and elastic_network() take that list.  The elastic
network joins only beads of the same domain, so it never joins two chains, nor two domains of one chain: it holds
each folded structure on its own.  Springs between chains (a complex) are not part of the model here; they would
be extra forces added to the System.  read_surfaces() reads the surface of every residue from one file, or from
one file per chain ("-" for a chain without folded domain).
"""

import math

import numpy as np
import openmm as mm

#                 mass     charge  radius  S_ref
RESIDUES = {
    'ALA': (71.079, 0.0, 0.2845, 0.796),
    'ARG': (157.197, 1.0, 0.3567, 1.921),
    'ASN': (114.104, 0.0, 0.3150, 1.281),
    'ASP': (114.080, -1.0, 0.3114, 1.162),
    'CYS': (103.139, 0.0, 0.3024, 1.074),
    'GLN': (128.131, 0.0, 0.3311, 1.575),
    'GLU': (128.107, -1.0, 0.3279, 1.462),
    'GLY': (57.052, 0.0, 0.2617, 0.544),
    'HIS': (137.142, 0.0, 0.3338, 1.634),
    'ILE': (113.160, 0.0, 0.3360, 1.410),
    'LEU': (113.160, 0.0, 0.3363, 1.519),
    'LYS': (129.183, 1.0, 0.3439, 1.923),
    'MET': (131.193, 0.0, 0.3381, 1.620),
    'PHE': (147.177, 0.0, 0.3556, 1.869),
    'PRO': (98.125, 0.0, 0.3187, 0.974),
    'SER': (87.078, 0.0, 0.2927, 0.933),
    'THR': (101.105, 0.0, 0.3108, 1.128),
    'TRP': (186.214, 0.0, 0.3754, 2.227),
    'TYR': (163.176, 0.0, 0.3611, 2.018),
    'VAL': (99.133, 0.0, 0.3205, 1.232),
}
POLAR = {'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LYS', 'SER', 'THR'}
CATIONIC = {'ARG', 'LYS'}
AROMATIC = {'PHE', 'TRP', 'TYR'}

EPS_POLAR, EPS_HYDROPHOBIC = 0.176, 0.295       # kJ/mol
A0_POLAR, A0_HYDROPHOBIC = 0.0, 0.0002          # kJ nm/mol
EPS_CATION_PI, EPS_PI_PI = 0.30, 0.10           # kJ/mol
BOND_LENGTH, BOND_K = 0.38, 4184.0              # nm, kJ/mol/nm^2
ANGLE_K = 4.184                                 # kJ/mol/rad^2
ENM_K, ENM_CUTOFF = 500.0, 0.9                  # kJ/mol/nm^2, nm
KAPPA = 1.0                                     # nm
CUTOFF = 3.0                                    # nm
LAMBDA = 0.7


def read_beads(path):
    """Read a coarse-grained PDB file with one bead per residue.

    Returns the residue names, the chain of each bead and the positions in nm (an N x 3 array).  The
    chain is the segment identifier (columns 73-76) if present, otherwise the chain identifier
    (column 22); consecutive beads of the same chain are bonded.
    """
    names, chains, positions = [], [], []
    with open(path) as pdb:
        for line in pdb:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            names.append(line[17:20].strip())
            chains.append(line[72:76].strip() or line[21])
            positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    unknown = sorted(set(names) - set(RESIDUES))
    if unknown:
        raise ValueError('residues not in the COCOMO2 table: %s' % ', '.join(unknown))
    return names, chains, 0.1*np.array(positions)


def exposure(names, sasa, domains, lam=LAMBDA):
    """Scaling factors xi of the nonbonded terms: 1 outside the folded domains, min(S/S_ref, lam)/lam
    inside.  sasa holds the solvent accessible surface of each residue (nm^2) in the reference
    structure, computed from the all-atom structure with atomic radii, as the reference surfaces S_ref of
    RESIDUES; domains is a list of (first, last) residue numbers, counted from 1, both included."""
    xi = np.ones(len(names))
    for first, last in domains:
        for i in range(first - 1, last):
            if np.isnan(sasa[i]):
                raise ValueError('residue %d lies in the folded domain %d-%d but has no surface (a chain given "-" '
                                 'in place of its surface file)' % (i + 1, first, last))
            xi[i] = min(sasa[i]/RESIDUES[names[i]][3], lam)/lam
    return xi


def elastic_network(positions, domains):
    """Pairs (i, j, r0) of the elastic network: beads of the same domain at least three residues apart
    and closer than 0.9 nm in the reference positions.  domains as in exposure()."""
    pairs = []
    for first, last in domains:
        for i in range(first - 1, last):
            for j in range(i + 3, last):
                r0 = float(np.linalg.norm(positions[j] - positions[i]))
                if r0 < ENM_CUTOFF:
                    pairs.append((i, j, r0))
    return pairs


def chain_ranges(chains):
    """The chains of a structure, in the order of the file, as (start, end) bead indices counted from 0, end
    excluded: a chain is a run of consecutive beads with the same chain identifier (read_beads()).  Two proteins
    that follow each other with the same identifier form one chain, bonded at the junction: give every protein its
    own identifier."""
    ranges = []
    for i in range(len(chains)):
        if i == 0 or chains[i] != chains[i - 1]:
            ranges.append([i, i + 1])
        else:
            ranges[-1][1] = i + 1
    return [tuple(r) for r in ranges]


def read_domains(path, chains):
    """The folded domains of every chain, from a text file with one line per chain, in the order of the chains in
    the structure (chain_ranges()), returned as a list of (first, last) in the numbering of the whole structure,
    counted from 1, both included, as exposure() and elastic_network() take them.

    On each line, the domains of the chain as pairs FIRST LAST separated by commas, in residue numbers counted from
    1 inside the chain (the position of the bead in its chain, not the residue number of the PDB file), both
    included, in increasing order and without overlaps; a line with only "-" is a chain without folded domain (a
    disordered protein: no exposure scaling and no elastic network).  Text after "#" is a comment; empty lines are
    skipped.  For example, for three chains of 100, 150 and 200 beads,

        10 12, 20 23     # chain 1: two domains
        2 10             # chain 2: one domain
        -                # chain 3: disordered

    gives [(10, 12), (20, 23), (102, 110)].  The file must have exactly one line per chain, and every domain must
    lie inside its chain; an error says where it does not."""
    ranges = chain_ranges(chains)
    lines = []
    with open(path) as text:
        for number, line in enumerate(text, start=1):
            line = line.split('#', 1)[0].strip()
            if line:
                lines.append((number, line))
    if len(lines) != len(ranges):
        raise ValueError('%s has %d lines with domains and the structure %d chains: give one line per chain, in the '
                         'order of the chains, and "-" for a chain without folded domain' % (
                             path, len(lines), len(ranges)))
    domains = []
    for (number, line), (start, end), chain in zip(lines, ranges, (chains[r[0]] for r in ranges)):
        if line == '-':
            continue
        previous = 0
        for pair in line.split(','):
            values = pair.split()
            if len(values) != 2 or not all(value.isdigit() for value in values):
                raise ValueError('%s, line %d: expected pairs of whole numbers FIRST LAST separated by commas, or "-", '
                                 'got %r' % (path, number, pair.strip()))
            first, last = int(values[0]), int(values[1])
            if not 1 <= first <= last <= end - start:
                raise ValueError('%s, line %d: the domain %d %d is not inside chain %s, which has %d residues' % (
                    path, number, first, last, chain, end - start))
            if first <= previous:
                raise ValueError('%s, line %d: the domain %d %d overlaps or precedes the one before it, which ends '
                                 'at %d' % (path, number, first, last, previous))
            previous = last
            domains.append((start + first, start + last))
    return domains


def read_surfaces(paths, chains):
    """The solvent accessible surface of every residue (nm^2), from one file for the whole structure, or from one
    file per chain, in the order of the chains (chain_ranges()), with "-" in place of the file of a chain without
    folded domain, whose surfaces are then undefined (NaN; exposure() refuses them in a domain).  Every file must
    hold one value per residue of its chain, or of the structure."""
    ranges = chain_ranges(chains)
    if len(paths) == 1 and paths[0] != '-' and len(ranges) > 1:
        sasa = np.atleast_1d(np.loadtxt(paths[0]))
        if len(sasa) != len(chains):
            raise ValueError('%s holds %d values and the structure has %d beads: give one value per residue of the '
                             'structure, or one file per chain' % (paths[0], len(sasa), len(chains)))
        return sasa
    if len(paths) != len(ranges):
        raise ValueError('%d surface files for %d chains: give one file for the whole structure, or one per chain in '
                         'the order of the chains, with "-" for a chain without folded domain' % (
                             len(paths), len(ranges)))
    parts = []
    for k, (path, (start, end)) in enumerate(zip(paths, ranges)):
        if path == '-':
            parts.append(np.full(end - start, np.nan))
            continue
        values = np.atleast_1d(np.loadtxt(path))
        if len(values) != end - start:
            raise ValueError('%s holds %d values and chain %d (%s) has %d residues' % (
                path, len(values), k + 1, chains[start], end - start))
        parts.append(values)
    return np.concatenate(parts)


def folded_domains(chains, domain=None, enm_domain=None, domains=None, enm_domains=None):
    """The folded domains for the exposure scaling and for the elastic network, from the options of the scripts:
    one range of the whole structure (--domain, --enm-domain: (first, last) counted from 1) or a file with the
    domains of every chain (--domains, --enm-domains: read_domains()).  The elastic network takes the domains of
    the exposure unless it has its own; without a domain or domains file there is no exposure scaling.  The
    scripts fill in the default ranges of their structure before calling it (examples/README.md, Folded domains
    in the scripts).  Returns two lists of (first, last)."""
    exposed = read_domains(domains, chains) if domains else ([tuple(domain)] if domain else [])
    if enm_domains:
        network = read_domains(enm_domains, chains)
    elif enm_domain:
        network = [tuple(enm_domain)]
    else:
        network = exposed
    return exposed, network


def _pair_force(expression, names, xi, epsilon, cutoff, bonds):
    force = mm.CustomNonbondedForce(expression + '; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); '
                                    'xi=sqrt(xi1*xi2)')
    for parameter in ('sigma', 'epsilon', 'xi'):
        force.addPerParticleParameter(parameter)
    for i, name in enumerate(names):
        force.addParticle([2*RESIDUES[name][2]*2**(-1/6), epsilon(name), xi[i]])
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)
    force.createExclusionsFromBonds(bonds, 1)
    return force


def create_system(names, chains, box, xi=None, enm_pairs=(), cutoff=CUTOFF):
    """An OpenMM System with the COCOMO2 model, in a cubic periodic box of side box (nm).

    names and chains come from read_beads(); xi from exposure() (default: 1 for every bead, a
    disordered protein); enm_pairs from elastic_network().
    """
    n = len(names)
    xi = np.ones(n) if xi is None else np.asarray(xi)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    for name in names:
        system.addParticle(RESIDUES[name][0])

    bonds = [(i, i + 1) for i in range(n - 1) if chains[i] == chains[i + 1]]
    bondForce = mm.HarmonicBondForce()
    for i, j in bonds:
        bondForce.addBond(i, j, BOND_LENGTH, BOND_K)
    system.addForce(bondForce)
    angleForce = mm.HarmonicAngleForce()
    for i in range(n - 2):
        if chains[i] == chains[i + 1] == chains[i + 2]:
            angleForce.addAngle(i, i + 1, i + 2, math.pi, ANGLE_K)
    system.addForce(angleForce)

    electrostatic = mm.CustomNonbondedForce('xi*(A1*A2 + Z1 + Z2)*exp(-r/kappa)/r; xi=sqrt(xi1*xi2)')
    electrostatic.addGlobalParameter('kappa', KAPPA)
    for parameter in ('A', 'Z', 'xi'):
        electrostatic.addPerParticleParameter(parameter)
    for i, name in enumerate(names):
        q = RESIDUES[name][1]
        electrostatic.addParticle([math.copysign(math.sqrt(0.75*abs(q)), q),
                                   A0_POLAR if name in POLAR else A0_HYDROPHOBIC, xi[i]])
    electrostatic.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    electrostatic.setCutoffDistance(cutoff)
    electrostatic.createExclusionsFromBonds(bonds, 1)
    system.addForce(electrostatic)

    shortRange = 'xi*4*epsilon*((sigma/r)^10 - (sigma/r)^5)'
    system.addForce(_pair_force(shortRange, names, xi, lambda name: EPS_POLAR if name in POLAR else EPS_HYDROPHOBIC,
                                cutoff, bonds))
    cationic = [i for i, name in enumerate(names) if name in CATIONIC]
    aromatic = [i for i, name in enumerate(names) if name in AROMATIC]
    if cationic and aromatic:
        force = _pair_force(shortRange, names, xi, lambda name: EPS_CATION_PI, cutoff, bonds)
        force.addInteractionGroup(cationic, aromatic)
        system.addForce(force)
    if aromatic:
        force = _pair_force(shortRange, names, xi, lambda name: EPS_PI_PI, cutoff, bonds)
        force.addInteractionGroup(aromatic, aromatic)
        system.addForce(force)

    if enm_pairs:
        enm = mm.HarmonicBondForce()
        for i, j, r0 in enm_pairs:
            enm.addBond(i, j, r0, ENM_K)
        system.addForce(enm)
    return system


def centred(positions, box):
    """The positions translated so that their mean is at the centre of the box."""
    return positions - positions.mean(axis=0) + 0.5*box
