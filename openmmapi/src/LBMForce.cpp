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
#include <iostream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

LBMForce::LBMForce() : nx(0), ny(0), nz(0), randomNumberSeed(0), momentumRemovalFrequency(1), machCheckFrequency(100),
        density(602.214), viscosity(1.0035), friction(1.0), temperature(300.0), machNumberLimit(0.3),
        couplingScheme(EulerMaruyama), dragScheme(Explicit), fluidFluctuations(false), bodyAcceleration(0, 0, 0),
        initialVelocity(0, 0, 0) {
}

LBMForce::CouplingScheme LBMForce::getCouplingScheme() const {
    return couplingScheme;
}

void LBMForce::setCouplingScheme(CouplingScheme scheme) {
    couplingScheme = scheme;
}

LBMForce::DragScheme LBMForce::getDragScheme() const {
    return dragScheme;
}

void LBMForce::setDragScheme(DragScheme scheme) {
    dragScheme = scheme;
}

bool LBMForce::getFluidFluctuations() const {
    return fluidFluctuations;
}

void LBMForce::setFluidFluctuations(bool fluctuations) {
    fluidFluctuations = fluctuations;
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

int LBMForce::getMachCheckFrequency() const {
    return machCheckFrequency;
}

void LBMForce::setMachCheckFrequency(int frequency) {
    machCheckFrequency = frequency;
}

double LBMForce::getMachNumberLimit() const {
    return machNumberLimit;
}

void LBMForce::setMachNumberLimit(double limit) {
    machNumberLimit = limit;
}

void LBMForce::getSolidNodes(vector<int>& nodes) const {
    nodes = solidNodes;
}

void LBMForce::setSolidNodes(const vector<int>& nodes) {
    solidNodes = nodes;
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

void LBMForce::createCheckpoint(Context& context, ostream& stream) const {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).createCheckpoint(getContextImpl(context), stream);
}

void LBMForce::loadCheckpoint(Context& context, istream& stream) {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).loadCheckpoint(getContextImpl(context), stream);
}

double LBMForce::getFluidMachNumber(Context& context) const {
    return dynamic_cast<LBMForceImpl&>(getImplInContext(context)).getFluidMachNumber(getContextImpl(context));
}

Vec3 LBMForce::getWallForce(Context& context) const {
    return dynamic_cast<LBMForceImpl&>(getImplInContext(context)).getWallForce(getContextImpl(context));
}

void LBMForce::getLatticeParametersInContext(const Context& context, double& dx, double& dt, double& tau) const {
    dynamic_cast<const LBMForceImpl&>(getImplInContext(context)).getLatticeParameters(dx, dt, tau);
}

ForceImpl* LBMForce::createImpl() const {
    return new LBMForceImpl(*this);
}

void LBMForce::updateParametersInContext(Context& context) {
    dynamic_cast<LBMForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
