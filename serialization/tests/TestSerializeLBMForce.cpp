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
