//
// Created by salmon on 17-9-5.
//
#include "TimeIntegrator.h"
#include "Domain.h"
#include "Atlas.h"
#include "simpla/data/DataNode.h"
namespace simpla {
namespace engine {
struct TimeIntegrator::pimpl_s {
    Real m_time_now_ = 0.0;
    Real m_time_end_ = 1.0;
    Real m_time_step_ = 1.0;
    size_type m_step_ = 0;
};
TimeIntegrator::TimeIntegrator() : m_pimpl_(new pimpl_s) {}
TimeIntegrator::~TimeIntegrator() { delete m_pimpl_; }
Real TimeIntegrator::GetTimeNow() const { return m_pimpl_->m_time_now_; }
void TimeIntegrator::SetTimeNow(Real t) { m_pimpl_->m_time_now_ = t; }
Real TimeIntegrator::GetTimeEnd() const { return m_pimpl_->m_time_end_; }
void TimeIntegrator::SetTimeEnd(Real t) { m_pimpl_->m_time_end_ = t; }
Real TimeIntegrator::GetTimeStep() const { return m_pimpl_->m_time_step_; }
void TimeIntegrator::SetTimeStep(Real t) { m_pimpl_->m_time_step_ = t; }
size_type TimeIntegrator::GetStep() const { return m_pimpl_->m_step_; }
void TimeIntegrator::SetStep(size_type s) { m_pimpl_->m_step_ = s; }

std::shared_ptr<data::DataNode> TimeIntegrator::Serialize() const { return base_type::Serialize(); }

void TimeIntegrator::Deserialize(std::shared_ptr<data::DataNode> const &tdb) { base_type::Deserialize(tdb); }

void TimeIntegrator::InitialCondition(Real time_now) {
    for (auto &d : GetDomains()) { d.second->InitialCondition(time_now); }
}
void TimeIntegrator::BoundaryCondition(Real time_now, Real dt) {
    for (auto &d : GetDomains()) { d.second->BoundaryCondition(time_now, dt); }
}

void TimeIntegrator::ComputeFluxes(Real time_now, Real dt) {
    for (auto &d : GetDomains()) { d.second->ComputeFluxes(time_now, dt); }
}
Real TimeIntegrator::ComputeStableDtOnPatch(Real time_now, Real time_dt) {
    for (auto &d : GetDomains()) { time_dt = std::min(time_dt, d.second->ComputeStableDtOnPatch(time_now, time_dt)); }
    return time_dt;
}

void TimeIntegrator::Advance(Real time_now, Real time_dt) {
    Update();
    GetAtlas()->Foreach([&](std::shared_ptr<engine::MeshBlock> const &blk) {
        VERBOSE << " Block : " << blk->IndexBox();
        auto p = GetPatch(blk->GetGUID());

        for (auto &item : GetDomains()) {
            VERBOSE << "  Advance Domain : " << item.first;
            if (item.second->Push(blk, p)) {
                item.second->Advance(time_now, time_dt);
                p = item.second->Pop();
            };
        }
        SetPatch(blk->GetGUID(), p);
    });
}
void TimeIntegrator::CheckPoint() const {}
void TimeIntegrator::DoSetUp() {
    SetStep(db()->GetValue<size_type>("Step", m_pimpl_->m_step_));
    SetMaxStep(db()->GetValue<size_type>("MaxStep", 100));
    base_type::DoSetUp();
}
void TimeIntegrator::DoTearDown() { base_type::DoTearDown(); }
void TimeIntegrator::Run() {
    Update();
    while (!Done()) {
        VERBOSE << " [ STEP:" << std::setw(5) << GetStep() << " START ] ";
        if (GetStep() == 0) { CheckPoint(); }
        Synchronize();
        NextStep();
        if (GetCheckPointInterval() > 0 && GetStep() % GetCheckPointInterval() == 0) { CheckPoint(); };
        if (GetDumpInterval() > 0 && GetStep() % GetDumpInterval() == 0) { Dump(); };

        VERBOSE << " [ STEP:" << std::setw(5) << GetStep() - 1 << " STOP  ] ";
    }
}
void TimeIntegrator::NextStep() {
    Advance(GetTimeNow(), GetTimeStep());
    m_pimpl_->m_time_now_ += m_pimpl_->m_step_;
    ++m_pimpl_->m_step_;
}

void TimeIntegrator::Synchronize() {}
bool TimeIntegrator::Done() const { return GetTimeNow() >= GetTimeEnd() || GetStep() >= GetMaxStep(); }
}
}