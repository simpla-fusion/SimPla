//
// Created by salmon on 17-9-5.
//
#include "SimpleTimeIntegrator.h"
namespace simpla {
void SimpleTimeIntegrator::DoInitialize() { base_type::DoInitialize(); }
void SimpleTimeIntegrator::DoFinalize() { base_type::DoFinalize(); }
void SimpleTimeIntegrator::DoUpdate() { base_type::DoUpdate(); }
void SimpleTimeIntegrator::DoTearDown() {}

void SimpleTimeIntegrator::Synchronize() {}
Real SimpleTimeIntegrator::Advance(Real time_now, Real time_dt) {}
bool SimpleTimeIntegrator::Done() const {}

void SimpleTimeIntegrator::CheckPoint() const {}
void SimpleTimeIntegrator::Dump() const {}
}  // namespace simpla