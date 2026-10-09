/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForceProxy.h"
#include "LBMForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

LBMForceProxy::LBMForceProxy() : SerializationProxy("LBMForce") {
}

void LBMForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 9);
    const LBMForce& force = *reinterpret_cast<const LBMForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    int nx, ny, nz;
    force.getGridSize(nx, ny, nz);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    int px, py, pz;
    force.getDomainDecomposition(px, py, pz);
    node.setIntProperty("domainsX", px);
    node.setIntProperty("domainsY", py);
    node.setIntProperty("domainsZ", pz);
    node.setBoolProperty("particleCopiesCheck", force.getParticleCopiesCheck());
    node.setBoolProperty("densityHaloExchange", force.getDensityHaloExchange());
    node.setBoolProperty("velocityHaloExchange", force.getVelocityHaloExchange());
    node.setDoubleProperty("density", force.getFluidDensity());
    node.setDoubleProperty("viscosity", force.getKinematicViscosity());
    node.setDoubleProperty("friction", force.getFriction());
    node.setDoubleProperty("temperature", force.getTemperature());
    node.setIntProperty("randomSeed", force.getRandomNumberSeed());
    node.setIntProperty("momentumRemovalFrequency", force.getFluidMomentumRemovalFrequency());
    node.setIntProperty("machCheckFrequency", force.getMachCheckFrequency());
    node.setIntProperty("couplingScheme", force.getCouplingScheme());
    node.setIntProperty("dragScheme", force.getDragScheme());
    node.setIntProperty("interpolationStencil", force.getInterpolationStencil());
    node.setBoolProperty("fluidFluctuations", force.getFluidFluctuations());
    node.setIntProperty("wallScheme", force.getWallScheme());
    node.setDoubleProperty("machNumberLimit", force.getMachNumberLimit());
    Vec3 g = force.getBodyAcceleration();
    node.createChildNode("BodyAcceleration").setDoubleProperty("x", g[0]).setDoubleProperty("y", g[1]).setDoubleProperty("z", g[2]);
    Vec3 u = force.getInitialFluidVelocity();
    node.createChildNode("InitialFluidVelocity").setDoubleProperty("x", u[0]).setDoubleProperty("y", u[1]).setDoubleProperty("z", u[2]);
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++)
        particles.createChildNode("Particle").setIntProperty("index", force.getParticle(i));
    vector<int> solidNodes;
    force.getSolidNodes(solidNodes);
    SerializationNode& solid = node.createChildNode("SolidNodes");
    for (int index : solidNodes)
        solid.createChildNode("Node").setIntProperty("index", index);
    SerializationNode& faces = node.createChildNode("Faces");
    for (int face = 0; face < 6; face++) {
        Vec3 v = force.getFaceVelocity((LBMForce::Face) face);
        faces.createChildNode("Face").setIntProperty("type", force.getFaceBoundary((LBMForce::Face) face))
             .setDoubleProperty("vx", v[0]).setDoubleProperty("vy", v[1]).setDoubleProperty("vz", v[2])
             .setDoubleProperty("density", force.getFaceDensity((LBMForce::Face) face));
    }
}

void* LBMForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 9)
        throw OpenMMException("Unsupported version number");
    LBMForce* force = new LBMForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setGridSize(node.getIntProperty("nx"), node.getIntProperty("ny"), node.getIntProperty("nz"));
        force->setFluidDensity(node.getDoubleProperty("density"));
        force->setKinematicViscosity(node.getDoubleProperty("viscosity"));
        force->setFriction(node.getDoubleProperty("friction"));
        force->setTemperature(node.getDoubleProperty("temperature"));
        force->setRandomNumberSeed(node.getIntProperty("randomSeed"));
        force->setFluidMomentumRemovalFrequency(node.getIntProperty("momentumRemovalFrequency"));
        if (version >= 2) {
            force->setMachCheckFrequency(node.getIntProperty("machCheckFrequency"));
            force->setMachNumberLimit(node.getDoubleProperty("machNumberLimit"));
        }
        if (version >= 3)
            force->setCouplingScheme((LBMForce::CouplingScheme) node.getIntProperty("couplingScheme"));
        // Versions 1 to 3 were written before the drag scheme existed, with the explicit drag.
        if (version >= 4)
            force->setDragScheme((LBMForce::DragScheme) node.getIntProperty("dragScheme"));
        // Versions 1 to 8 were written before the interpolation stencils existed, with the nearest node.
        if (version >= 9)
            force->setInterpolationStencil((LBMForce::InterpolationStencil) node.getIntProperty("interpolationStencil"));
        // Versions 1 to 4 were written before the fluid fluctuations existed, without them.
        if (version >= 5)
            force->setFluidFluctuations(node.getBoolProperty("fluidFluctuations"));
        // Versions 1 to 7 were written before the domain decomposition existed, with one domain.
        if (version >= 8) {
            force->setDomainDecomposition(node.getIntProperty("domainsX"), node.getIntProperty("domainsY"),
                    node.getIntProperty("domainsZ"));
            force->setParticleCopiesCheck(node.getBoolProperty("particleCopiesCheck", true));
            force->setDensityHaloExchange(node.getBoolProperty("densityHaloExchange", false));
            force->setVelocityHaloExchange(node.getBoolProperty("velocityHaloExchange", false));
        }
        // Versions 1 to 5 were written before the wall schemes existed, with bounce-back.
        if (version >= 6)
            force->setWallScheme((LBMForce::WallScheme) node.getIntProperty("wallScheme"));
        const SerializationNode& g = node.getChildNode("BodyAcceleration");
        force->setBodyAcceleration(Vec3(g.getDoubleProperty("x"), g.getDoubleProperty("y"), g.getDoubleProperty("z")));
        const SerializationNode& u = node.getChildNode("InitialFluidVelocity");
        force->setInitialFluidVelocity(Vec3(u.getDoubleProperty("x"), u.getDoubleProperty("y"), u.getDoubleProperty("z")));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (const SerializationNode& particle : particles.getChildren())
            force->addParticle(particle.getIntProperty("index"));
        if (version >= 2) {
            vector<int> solidNodes;
            for (const SerializationNode& solid : node.getChildNode("SolidNodes").getChildren())
                solidNodes.push_back(solid.getIntProperty("index"));
            force->setSolidNodes(solidNodes);
        }
        // Versions 1 to 6 were written before the open faces existed, with a periodic box.
        if (version >= 7) {
            const vector<SerializationNode>& faces = node.getChildNode("Faces").getChildren();
            if (faces.size() != 6)
                throw OpenMMException("LBMForce: the serialized force must have 6 faces");
            for (int face = 0; face < 6; face++) {
                LBMForce::Face f = (LBMForce::Face) face;
                force->setFaceBoundary(f, (LBMForce::BoundaryType) faces[face].getIntProperty("type"));
                force->setFaceVelocity(f, Vec3(faces[face].getDoubleProperty("vx"), faces[face].getDoubleProperty("vy"),
                        faces[face].getDoubleProperty("vz")));
                force->setFaceDensity(f, faces[face].getDoubleProperty("density"));
            }
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
