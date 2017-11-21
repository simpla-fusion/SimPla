//
// Created by salmon on 17-9-5.
//
#include "TimeIntegrator.h"
#include "Atlas.h"
#include "Domain.h"
#include "simpla/data/DataEntry.h"
namespace simpla {
namespace engine {

TimeIntegrator::TimeIntegrator() = default;
TimeIntegrator::~TimeIntegrator() = default;

void TimeIntegrator::InitialCondition(Real time_now) {
    Update();
    GetAtlas()->Foreach([&](std::shared_ptr<Patch> const &patch) {
        if (patch == nullptr) { return; }

        for (auto &item : GetDomains()) {
            item.second->Push(patch->Pop());
            if (item.second->CheckBlockInBoundary() >= 0) { item.second->InitialCondition(time_now); }
            patch->Push(item.second->Pop());
        }

    });
}
void TimeIntegrator::BoundaryCondition(Real time_now, Real dt) {
    //    for (auto &d : GetDomains()) { d.second->BoundaryCondition(time_now, dt); }
}

void TimeIntegrator::ComputeFluxes(Real time_now, Real dt) {
    //    for (auto &d : GetDomains()) { d.second->ComputeFluxes(time_now, dt); }
}
Real TimeIntegrator::ComputeStableDtOnPatch(Real time_now, Real time_dt) {
    //    for (auto &d : GetDomains()) { time_dt = std::min(time_dt, d.second->ComputeStableDtOnPatch(time_now,
    //    time_dt)); }
    return time_dt;
}

void TimeIntegrator::Advance(Real time_now, Real time_dt) {
    Update();
    GetAtlas()->Foreach([&](std::shared_ptr<Patch> const &patch) {
        if (patch == nullptr) { return; }

        for (auto &item : GetDomains()) {
            item.second->Push(patch->Pop());

            if (item.second->CheckBlockInBoundary() >= 0) {
                if (item.second->IsInitialized()) { item.second->InitialCondition(time_now); }
                item.second->Advance(time_now, time_dt);
                if (item.second->CheckBlockInBoundary() == 0) { item.second->BoundaryCondition(time_now, time_dt); }
            }
            patch->Push(item.second->Pop());
        }
    });
}
void TimeIntegrator::DoSetUp() {
    //    SetStepNumber(backend()->GetValue<size_type>("Step", GetStepNumber()));
    //    SetTimeNow(
    //            backend()->GetValue<size_type>("MaxStep", static_cast<size_type>((GetTimeEnd() - GetTimeNow()) /
    //            GetTimeStep())));
    //    SetMaxStep(
    //        backend()->GetValue<size_type>("MaxStep", static_cast<size_type>((GetTimeEnd() - GetTimeNow()) /
    //        GetTimeStep())));
    //
    SetTimeStep((GetTimeEnd() - GetTimeNow()) / GetMaxStep());
    base_type::DoSetUp();
}
void TimeIntegrator::DoTearDown() { base_type::DoTearDown(); }
void TimeIntegrator::Run() {
    InitialCondition(GetTimeNow());
    Synchronize(0);
    CheckPoint(GetStepNumber());

    while (!Done()) {
        VERBOSE << " [ TIME :" << std::setw(5) << GetTimeNow() << "   ] ";
        Advance(GetTimeNow(), GetTimeStep());
        Synchronize(0);
        NextStep();
        CheckPoint(GetStepNumber());
    }

    //    Dump();
}
void TimeIntegrator::NextStep() {
    m_TimeNow_ += m_TimeStep_;
    base_type::NextStep();
}

void TimeIntegrator::Synchronize(int level) { base_type::Synchronize(level); }
bool TimeIntegrator::Done() const { return GetTimeNow() >= GetTimeEnd() || GetStepNumber() >= GetMaxStep(); }
}
}