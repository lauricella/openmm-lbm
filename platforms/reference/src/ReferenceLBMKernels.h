#ifndef REFERENCE_LBM_KERNELS_H_
#define REFERENCE_LBM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMKernels.h"
#include "openmm/Platform.h"
#include "sfmt/SFMT.h"
#include <vector>

namespace LBMPlugin {

/**
 * The Reference implementation of the lattice Boltzmann fluid and of its coupling to the particles: plain
 * loops in double precision.  It is the correctness reference for the other platforms.  The random numbers of the
 * particles and of the fluctuating fluid come from one generator owned by the kernel.
 *
 * One lattice step (advanceFluid) has the same structure as on the other platforms:
 *  1. moments: density, momentum, non-equilibrium second moment and body force at every node;
 *  2. removal of the fluid momentum, in the steps whose index (the step count of the Context) is a multiple
 *     of momentumRemovalFrequency;
 *  3. coupling: each coupled particle feels the drag (explicit or centred) and the random force at its nearest
 *     node, and the node receives the opposite force (docs/theory.md, section 2);
 *  4. collision and streaming: the populations are rebuilt from the moments of their own node and pushed
 *     to the neighbours.  The collision reads only moments, so a single population array is enough;
 *  5. the walls and the open faces: with bounce-back a population that streamed into a solid node is sent back to
 *     the fluid node it came from; then the populations of the boundary nodes (the fluid nodes next to the solid
 *     nodes with regularized walls, and those on open faces) are rebuilt.  Solid nodes have no moments and no
 *     collision.
 */
class ReferenceCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    ReferenceCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform) : CalcLBMForceKernel(name, platform),
            stepPending(false), stepForcesCurrent(false), stepIndex(0), machWarningPrinted(false), noiseDrawn(false), hasStoredGaussian(false),
            storedGaussian(0) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
    void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
private:
    void advanceFluid(OpenMM::ContextImpl& context);
    void computeNextStepForces(OpenMM::ContextImpl& context);
    void computeMoments();
    void removeFluidMomentum();
    void couple(OpenMM::ContextImpl& context, bool isStep);
    void coupleParticles(OpenMM::ContextImpl& context, bool isStep);
    void coupleParticlesCentered(OpenMM::ContextImpl& context, bool isStep);
    void drawNoise();
    void applyReaction();
    int nearestNode(const OpenMM::Vec3& position) const;
    OpenMM::Vec3 wallNormal(const OpenMM::Vec3& position) const;
    double getGaussianRandom();
    void collideAndStream();
    void bounceBack();
    void applyBoundaries();
    void checkMachNumber();
    double computeMachNumber() const;
    LBMLatticeParameters lattice;
    /** Deviations of the populations from the rest equilibrium at lattice density 1, f_q - w_q, stored as
        [q*numNodes + node].  They keep the precision of small signals, and they are the fluid state of
        getFluidState() and setFluidState(), so that saving and restoring the fluid is exact. */
    std::vector<double> populations;
    /** Deviation of the density from 1, rho - 1, at every node, from the moments of the current step. */
    std::vector<double> densityDeviation;
    /** Moments of the current step: density, momentum j = rho*u (3 per node), non-equilibrium second moment
        (xx, yy, zz, xy, xz, yz per node) and force density (3 per node), all in lattice units. */
    std::vector<double> rho, momentum, piNeq, forceDensity;
    /** 1 for fluid nodes, 0 for solid nodes; empty if there are no solid nodes. */
    std::vector<char> isFluid;
    /** The boundary nodes (internal/LBMBoundaries.h): with regularized walls the fluid nodes next to the solid
        nodes, which lie on the walls, and the fluid nodes on the open faces.  For each of them: the bits 1 << q of
        the directions q whose populations are unknown after the streaming (unknownDirections) and of those among
        them whose source node x - c_q is solid (solidDirections), its LBMBoundaries::Kind and the face that gives
        its velocity or density. */
    std::vector<int> boundaryNodes, unknownDirections, solidDirections, boundaryKind, boundaryFace;
    /** 1 for the nodes of regularized walls, 2 for the other boundary nodes (on open faces), 0 elsewhere; empty if
        there are no boundary nodes. */
    std::vector<char> isBoundary;
    /** True between beginStep() and the force evaluation of that integration step. */
    bool stepPending;
    /** True from the lattice step until the coupling forces are recomputed or the state is replaced: the forces
        hold those of the step. */
    bool stepForcesCurrent;
    /** Step count of the Context: set by beginStep() at the start of a lattice step, incremented at its end. */
    long long stepIndex;
    /** True once the debug warning about the Mach number has been printed. */
    bool machWarningPrinted;
    /** Masses of the coupled particles in lattice units, m/m_c. */
    std::vector<double> particleMass;
    /** Forces on the coupled particles (kJ/mol/nm): in the force evaluation of an integration step, the force of
        that step; in the evaluations between steps, the force of the next step (docs/theory.md, section 2). */
    std::vector<OpenMM::Vec3> particleForces;
    /** Random numbers xi of the next lattice step, three per coupled particle, drawn once per step by the first
        evaluation that needs them and used by the step itself. */
    std::vector<OpenMM::Vec3> noise;
    /** True if noise holds the random numbers of the next step. */
    bool noiseDrawn;
    /** Momentum (Da nm/ps) given to the solid nodes in the current or last lattice step. */
    OpenMM::Vec3 wallMomentum;
    /** Reaction of the coupled particles on each node (3 per node), in lattice units. */
    std::vector<double> reaction;
    /** Generator of the random force and of the fluctuations of the fluid, owned by the kernel so that its sequence
        does not depend on other forces.  In a step the fluid draws its numbers after the coupling, so the sequence
        does not depend on the force evaluations between steps either. */
    OpenMM_SFMT::SFMT sfmt;
    /** The Box-Muller transform yields two Gaussian numbers: the second is kept for the next call. */
    bool hasStoredGaussian;
    double storedGaussian;
};

} // namespace LBMPlugin

#endif /*REFERENCE_LBM_KERNELS_H_*/
