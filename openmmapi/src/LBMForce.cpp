/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForce.h"
#include "internal/LBMForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

LBMForce::LBMForce() : nx(0), ny(0), nz(0), randomNumberSeed(0), momentumRemovalFrequency(1),
        density(602.214), viscosity(1.0035), friction(1.0), temperature(300.0),
        bodyAcceleration(0, 0, 0), initialVelocity(0, 0, 0) {
}

void LBMForce::getGridSize(int& nx, int& ny, int& nz) const {
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void LBMForce::setGridSize(int nx, int ny, int nz) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

double LBMForce::getFluidDensity() const {
    return density;
}

void LBMForce::setFluidDensity(double density) {
    this->density = density;
}

double LBMForce::getKinematicViscosity() const {
    return viscosity;
}

void LBMForce::setKinematicViscosity(double viscosity) {
    this->viscosity = viscosity;
}

double LBMForce::getFriction() const {
    return friction;
}

void LBMForce::setFriction(double friction) {
    this->friction = friction;
}

double LBMForce::getTemperature() const {
    return temperature;
}

void LBMForce::setTemperature(double temperature) {
    this->temperature = temperature;
}

int LBMForce::getRandomNumberSeed() const {
    return randomNumberSeed;
}

void LBMForce::setRandomNumberSeed(int seed) {
    randomNumberSeed = seed;
}

Vec3 LBMForce::getBodyAcceleration() const {
    return bodyAcceleration;
}

void LBMForce::setBodyAcceleration(const Vec3& acceleration) {
    bodyAcceleration = acceleration;
}

Vec3 LBMForce::getInitialFluidVelocity() const {
    return initialVelocity;
}

void LBMForce::setInitialFluidVelocity(const Vec3& velocity) {
    initialVelocity = velocity;
}

int LBMForce::getFluidMomentumRemovalFrequency() const {
    return momentumRemovalFrequency;
}

void LBMForce::setFluidMomentumRemovalFrequency(int frequency) {
    momentumRemovalFrequency = frequency;
}

int LBMForce::addParticle(int particle) {
    particles.push_back(particle);
    return particles.size()-1;
}

int LBMForce::getParticle(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index];
}

void LBMForce::setParticle(int index, int particle) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index] = particle;
}

void LBMForce::getFluidFields(Context& context, vector<double>& density, vector<Vec3>& velocity) const {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).getFluidFields(getContextImpl(context), density, velocity);
}

void LBMForce::getFluidState(Context& context, vector<double>& state) const {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).getFluidState(getContextImpl(context), state);
}

void LBMForce::setFluidState(Context& context, const vector<double>& state) {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).setFluidState(getContextImpl(context), state);
}

ForceImpl* LBMForce::createImpl() const {
    return new LBMForceImpl(*this);
}

void LBMForce::updateParametersInContext(Context& context) {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
