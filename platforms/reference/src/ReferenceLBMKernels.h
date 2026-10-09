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
#include "internal/LBMDecomposition.h"
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
            storedGaussian(0), replicasChecked(false) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity, bool halo);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
    void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
    void createRankCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadRankCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
    void resetRankState(OpenMM::ContextImpl& context);
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
    void computeWallMomentum();
    void applyBoundaries();
    void checkMachNumber();
    double computeMachNumber() const;
    /** True if this rank advances the node (always without domain decomposition). */
    bool isOwned(int node) const {
        return owned.empty() || owned[node];
    }
    void exchangePopulations();
    /** With the domain decomposition: sum the coupling forces over the ranks, and check that the copies of the
        particles are the same on every rank (collective). */
    void sumParticleForces();
    void checkReplicas(OpenMM::ContextImpl& context);
    /** The density and velocity of a node in lattice units, from its populations. */
    void nodeFields(int node, double& density, OpenMM::Vec3& velocity) const;
    /** With the domain decomposition and the exchange of the halo: send the fields of the nodes of the rank to the
        ranks whose halo contains them, and receive those of the halo of the rank (collective). */
    void exchangeHalo();
    /** Collision and streaming of one node; normal holds its 15 normal numbers with a fluctuating fluid. */
    void collideNode(int node, double mu, const double* normal);
    LBMLatticeParameters lattice;
    /** The domain decomposition (docs/theory.md, Domain decomposition): the arrays cover the whole lattice on every
        rank, and each rank advances the nodes it owns (owned, empty without decomposition).  After the streaming
        it sends to rank r the populations it pushed into fluid nodes of r, at the slots q*numNodes + node of
        sendSlots[r], and receives those of its own nodes at receiveSlots[r]; both lists are in the order of
        (node, q).  The slots of solid nodes are neither sent nor received. */
    LBMDecomposition decomposition;
    std::vector<char> owned;
    std::vector<std::vector<int> > sendSlots, receiveSlots;
    /** With the decomposition the owned fluid nodes are advanced in two groups, the frame (nodes that push into
        other ranks) and the interior, with the exchange of the frame running meanwhile.  The normal numbers of a
        fluctuating fluid are drawn first, in node order (normalIndex gives the place of each node), so that the
        order of the groups does not change them. */
    std::vector<int> frameNodes, interiorNodes, normalIndex;
    std::vector<double> fluidNormals;
    std::vector<std::vector<double> > sendBuffers, receiveBuffers;
    /** The halo of the rank (setDensityHaloExchange(), setVelocityHaloExchange()): for each rank r, the nodes of this
        rank in the halo of r and the nodes of r in the halo of this rank, in index order; and the fields of the halo
        in lattice units (density, and velocity with 3 values per node, over the whole lattice, NaN outside the
        halo), empty when not exchanged. */
    std::vector<std::vector<int> > haloSendNodes, haloReceiveNodes;
    std::vector<double> haloDensity, haloVelocity;
    /** True once the copies of the particles have been compared (checkReplicas()). */
    bool replicasChecked;
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
    /** With bounce-back walls, the fluid nodes next to the solid nodes and, for each of them, the bits 1 << q of the
        directions q whose node x + c_q is solid (LBMBoundaries::findWallLinks()); empty otherwise. */
    std::vector<int> wallNodes, wallLinks;
    /** The boundary nodes (internal/LBMBoundaries.h): with regularized walls the fluid nodes next to the solid
        nodes (the walls lie on the solid nodes), and the fluid nodes on the open faces.  For each of them: the bits
        1 << q of the directions q whose populations are unknown after the streaming (unknownDirections) and of
        those among them whose source node x - c_q is solid (solidDirections), its LBMBoundaries::Kind and the face
        that gives its velocity or density. */
    std::vector<int> boundaryNodes, unknownDirections, solidDirections, boundaryKind, boundaryFace;
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
