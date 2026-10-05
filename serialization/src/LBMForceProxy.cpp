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
    node.setIntProperty("version", 2);
    const LBMForce& force = *reinterpret_cast<const LBMForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    int nx, ny, nz;
    force.getGridSize(nx, ny, nz);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    node.setDoubleProperty("density", force.getFluidDensity());
    node.setDoubleProperty("viscosity", force.getKinematicViscosity());
    node.setDoubleProperty("friction", force.getFriction());
    node.setDoubleProperty("temperature", force.getTemperature());
    node.setIntProperty("randomSeed", force.getRandomNumberSeed());
    node.setIntProperty("momentumRemovalFrequency", force.getFluidMomentumRemovalFrequency());
    node.setIntProperty("machCheckFrequency", force.getMachCheckFrequency());
    node.setDoubleProperty("machNumberLimit", force.getMachNumberLimit());
    Vec3 g = force.getBodyAcceleration();
    node.createChildNode("BodyAcceleration").setDoubleProperty("x", g[0]).setDoubleProperty("y", g[1]).setDoubleProperty("z", g[2]);
    Vec3 u = force.getInitialFluidVelocity();
    node.createChildNode("InitialFluidVelocity").setDoubleProperty("x", u[0]).setDoubleProperty("y", u[1]).setDoubleProperty("z", u[2]);
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++)
        particles.createChildNode("Particle").setIntProperty("index", force.getParticle(i));
}

void* LBMForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 2)
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
        const SerializationNode& g = node.getChildNode("BodyAcceleration");
        force->setBodyAcceleration(Vec3(g.getDoubleProperty("x"), g.getDoubleProperty("y"), g.getDoubleProperty("z")));
        const SerializationNode& u = node.getChildNode("InitialFluidVelocity");
        force->setInitialFluidVelocity(Vec3(u.getDoubleProperty("x"), u.getDoubleProperty("y"), u.getDoubleProperty("z")));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (const SerializationNode& particle : particles.getChildren())
            force->addParticle(particle.getIntProperty("index"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
