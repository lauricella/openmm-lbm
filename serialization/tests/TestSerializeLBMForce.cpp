/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerLBMSerializationProxies();

void testSerialization() {
    // Create a Force with values different from the defaults.

    LBMForce force;
    force.setForceGroup(3);
    force.setName("fluid");
    force.setGridSize(10, 12, 14);
    force.setFluidDensity(500.0);
    force.setKinematicViscosity(2.5);
    force.setFriction(5.0);
    force.setTemperature(310.0);
    force.setRandomNumberSeed(123);
    force.setFluidMomentumRemovalFrequency(10);
    force.setMachCheckFrequency(25);
    force.setMachNumberLimit(0.2);
    force.setCouplingScheme(LBMForce::NVE);
    force.setDragScheme(LBMForce::Centered);
    force.setFluidFluctuations(true);
    force.setSolidNodes(vector<int>({0, 7, 42}));
    force.setBodyAcceleration(Vec3(0.1, 0.2, 0.3));
    force.setInitialFluidVelocity(Vec3(-0.1, 0.0, 0.05));
    force.addParticle(3);
    force.addParticle(7);
    force.addParticle(1);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<LBMForce>(&force, "Force", buffer);
    LBMForce* copy = XmlSerializer::deserialize<LBMForce>(buffer);

    // Compare the two forces to see if they are identical.

    LBMForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    int nx, ny, nz, nx2, ny2, nz2;
    force.getGridSize(nx, ny, nz);
    force2.getGridSize(nx2, ny2, nz2);
    ASSERT_EQUAL(nx, nx2);
    ASSERT_EQUAL(ny, ny2);
    ASSERT_EQUAL(nz, nz2);
    ASSERT_EQUAL(force.getFluidDensity(), force2.getFluidDensity());
    ASSERT_EQUAL(force.getKinematicViscosity(), force2.getKinematicViscosity());
    ASSERT_EQUAL(force.getFriction(), force2.getFriction());
    ASSERT_EQUAL(force.getTemperature(), force2.getTemperature());
    ASSERT_EQUAL(force.getRandomNumberSeed(), force2.getRandomNumberSeed());
    ASSERT_EQUAL(force.getFluidMomentumRemovalFrequency(), force2.getFluidMomentumRemovalFrequency());
    ASSERT_EQUAL(force.getMachCheckFrequency(), force2.getMachCheckFrequency());
    ASSERT_EQUAL(force.getMachNumberLimit(), force2.getMachNumberLimit());
    ASSERT_EQUAL(force.getCouplingScheme(), force2.getCouplingScheme());
    ASSERT_EQUAL(force.getDragScheme(), force2.getDragScheme());
    ASSERT_EQUAL(force.getFluidFluctuations(), force2.getFluidFluctuations());
    vector<int> solid1, solid2;
    force.getSolidNodes(solid1);
    force2.getSolidNodes(solid2);
    ASSERT_EQUAL_CONTAINERS(solid1, solid2);
    ASSERT_EQUAL_VEC(force.getBodyAcceleration(), force2.getBodyAcceleration(), 0.0);
    ASSERT_EQUAL_VEC(force.getInitialFluidVelocity(), force2.getInitialFluidVelocity(), 0.0);
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++)
        ASSERT_EQUAL(force.getParticle(i), force2.getParticle(i));
    delete copy;

    // A force written before the fluid fluctuations existed (version 4) has none, and one written before the drag
    // scheme existed (version 3) has the explicit drag.

    string xml = buffer.str();
    size_t fluctuations = xml.find(" fluidFluctuations=\"1\"");
    ASSERT(fluctuations != string::npos);
    xml.erase(fluctuations, 22);
    size_t version = xml.find("version=\"5\"");
    ASSERT(version != string::npos);
    xml.replace(version, 11, "version=\"4\"");
    stringstream buffer4(xml);
    LBMForce* copy4 = XmlSerializer::deserialize<LBMForce>(buffer4);
    ASSERT(!copy4->getFluidFluctuations());
    ASSERT_EQUAL(LBMForce::Centered, copy4->getDragScheme());
    delete copy4;
    size_t drag = xml.find(" dragScheme=\"1\"");
    ASSERT(drag != string::npos);
    xml.erase(drag, 15);
    version = xml.find("version=\"4\"");
    ASSERT(version != string::npos);
    xml.replace(version, 11, "version=\"3\"");
    stringstream oldBuffer(xml);
    LBMForce* oldCopy = XmlSerializer::deserialize<LBMForce>(oldBuffer);
    ASSERT_EQUAL(LBMForce::Explicit, oldCopy->getDragScheme());
    ASSERT(!oldCopy->getFluidFluctuations());
    ASSERT_EQUAL(force.getCouplingScheme(), oldCopy->getCouplingScheme());
    delete oldCopy;
}

int main() {
    try {
        registerLBMSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
