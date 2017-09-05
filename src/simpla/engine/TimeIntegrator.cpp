//
// Created by salmon on 17-9-5.
//
#include "TimeIntegrator.h"
#include "Domain.h"
#include "Mesh.h"
#include "simpla/data/DataNode.h"
namespace simpla {
namespace engine {

TimeIntegrator::TimeIntegrator() {}
TimeIntegrator::~TimeIntegrator() {}

std::shared_ptr<data::DataNode> TimeIntegrator::Serialize() const { return base_type::Serialize(); }

void TimeIntegrator::Deserialize(std::shared_ptr<data::DataNode> const &tdb) { base_type::Deserialize(tdb); }

void TimeIntegrator::InitialCondition(Real time_now) {
    GetMesh()->InitialCondition(time_now);
    for (auto &d : GetDomains()) { d.second->InitialCondition(time_now); }
}
void TimeIntegrator::BoundaryCondition(Real time_now, Real dt) {
    GetMesh()->BoundaryCondition(time_now, dt);
    for (auto &d : GetDomains()) { d.second->BoundaryCondition(time_now, dt); }
}

void TimeIntegrator::ComputeFluxes(Real time_now, Real dt) {
    for (auto &d : GetDomains()) { d.second->ComputeFluxes(time_now, dt); }
}
Real TimeIntegrator::ComputeStableDtOnPatch(Real time_now, Real time_dt) {
    for (auto &d : GetDomains()) { time_dt = d.second->ComputeStableDtOnPatch(time_now, time_dt); }
    return time_dt;
}

void TimeIntegrator::Advance(Real time_now, Real dt) {
    GetMesh()->Advance(time_now, dt);
    for (auto &d : GetDomains()) { d.second->Advance(time_now, dt); }
}
void TimeIntegrator::CheckPoint() const {}

void TimeIntegrator::Run() {
    while (!Done()) {
        VERBOSE << " [ STEP:" << std::setw(5) << GetStep() << " START ] " << std::endl;
        if (GetStep() == 0) { CheckPoint(); }
        Synchronize();
        NextStep();
        if (GetCheckPointInterval() > 0 && GetStep() % GetCheckPointInterval() == 0) { CheckPoint(); };
        if (GetDumpInterval() > 0 && GetStep() % GetDumpInterval() == 0) { Dump(); };

        VERBOSE << " [ STEP:" << std::setw(5) << GetStep() - 1 << " STOP  ] " << std::endl;
    }
}
void TimeIntegrator::NextStep() {
    Advance(GetTimeNow(), GetTimeStep());
    SetTimeNow(GetTimeNow() + GetTimeStep());
    SetStep(GetStep() + 1);
}

void TimeIntegrator::Synchronize() {}
bool TimeIntegrator::Done() const { return GetStep() >= GetMaxStep(); }
}
}