//
// Created by salmon on 17-9-5.
//
#include "TimeIntegrator.h"
#include "Atlas.h"
#include "Domain.h"
#include "simpla/data/DataNode.h"
namespace simpla {
namespace engine {
struct TimeIntegrator::pimpl_s {
    Real m_time_now_ = 0.0;
    Real m_time_end_ = 1.0;
    Real m_time_step_ = 1.0;
};
TimeIntegrator::TimeIntegrator() : m_pimpl_(new pimpl_s) {}
TimeIntegrator::~TimeIntegrator() { delete m_pimpl_; }
Real TimeIntegrator::GetTimeNow() const { return m_pimpl_->m_time_now_; }
void TimeIntegrator::SetTimeNow(Real t) { m_pimpl_->m_time_now_ = t; }
Real TimeIntegrator::GetTimeEnd() const { return m_pimpl_->m_time_end_; }
void TimeIntegrator::SetTimeEnd(Real t) { m_pimpl_->m_time_end_ = t; }
Real TimeIntegrator::GetTimeStep() const { return m_pimpl_->m_time_step_; }
void TimeIntegrator::SetTimeStep(Real t) { m_pimpl_->m_time_step_ = t; }

std::shared_ptr<data::DataNode> TimeIntegrator::Serialize() const { return base_type::Serialize(); }
void TimeIntegrator::Deserialize(std::shared_ptr<data::DataNode> const &tdb) { base_type::Deserialize(tdb); }

void TimeIntegrator::InitialCondition(Real time_now) {
    Update();
    GetAtlas()->Foreach([&](std::shared_ptr<Patch> const &patch) {
        if (patch == nullptr) { return; }

        for (auto &item : GetDomains()) {
            item.second->Push(patch->Pop());
            if (item.second->CheckBoundary() >= 0) { item.second->InitialCondition(time_now); }
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

            if (item.second->CheckBoundary() >= 0) {
                if (item.second->IsInitialized()) { item.second->InitialCondition(time_now); }
                item.second->Advance(time_now, time_dt);
                if (item.second->CheckBoundary() == 0) { item.second->BoundaryCondition(time_now, time_dt); }
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
    SetTimeStep(db()->GetValue<Real>("TimeStep", (GetTimeEnd() - GetTimeNow()) / GetMaxStep()));
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
    m_pimpl_->m_time_now_ += m_pimpl_->m_time_step_;
    base_type::NextStep();
}

void TimeIntegrator::Synchronize(int level) { base_type::Synchronize(level); }
bool TimeIntegrator::Done() const { return GetTimeNow() >= GetTimeEnd() || GetStepNumber() >= GetMaxStep(); }
}
}