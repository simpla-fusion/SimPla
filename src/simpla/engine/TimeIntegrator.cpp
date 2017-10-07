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
    //    for (auto &d : GetDomains()) { d.second->InitialCondition(time_now); }

    Update();
    GetAtlas()->Foreach([&](std::shared_ptr<engine::MeshBlock> const &blk) {
        ASSERT(blk != nullptr);
        VERBOSE << std::setw(20) << "Block : " << blk->GetIndexBox();
        auto patch = GetPatch(blk->GetGUID());
        auto out_patch = data::DataNode::New(data::DataNode::DN_TABLE);
        if (patch == nullptr) { patch = data::DataNode::New(data::DataNode::DN_TABLE); }

        for (auto &item : GetDomains()) {
            item.second->SetBlock(blk);
            int chk_bdry = item.second->CheckBoundary();
            if (chk_bdry == DomainBase::OUT_BOUNDARY) { continue; }
            item.second->Push(patch);
            item.second->InitialCondition(time_now);
            patch->Set(item.second->Pop());
        }

        SetPatch(blk->GetGUID(), patch);

    });
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
        ASSERT(blk != nullptr);
        VERBOSE << std::setw(20) << "Block : " << blk->GetIndexBox();
        auto patch = GetPatch(blk->GetGUID());
        auto out_patch = data::DataNode::New(data::DataNode::DN_TABLE);
        bool need_init_cond = false;
        if (patch == nullptr) {
            patch = data::DataNode::New(data::DataNode::DN_TABLE);
            need_init_cond = true;
        }

        for (auto &item : GetDomains()) {
            item.second->SetBlock(blk);
            int chk_bdry = item.second->CheckBoundary();
            if (chk_bdry == DomainBase::OUT_BOUNDARY) { continue; }
            item.second->Push(patch);
            if (need_init_cond) { item.second->InitialCondition(time_now); }
            item.second->Advance(time_now, time_dt);
            if (chk_bdry == DomainBase::ON_BOUNDARY) { item.second->BoundaryCondition(time_now, time_dt); }
            patch->Set(item.second->Pop());
        }

        SetPatch(blk->GetGUID(), patch);

    });
}
void TimeIntegrator::DoSetUp() {
    //    SetStepNumber(db()->GetValue<size_type>("Step", GetStepNumber()));
    //    SetTimeNow(
    //            db()->GetValue<size_type>("MaxStep", static_cast<size_type>((GetTimeEnd() - GetTimeNow()) /
    //            GetTimeStep())));
    //    SetMaxStep(
    //        db()->GetValue<size_type>("MaxStep", static_cast<size_type>((GetTimeEnd() - GetTimeNow()) /
    //        GetTimeStep())));
    //
    SetTimeStep(db()->GetValue<Real>("TimeStep", (GetTimeEnd() - GetTimeNow()) / GetMaxStep()));
    base_type::DoSetUp();
}
void TimeIntegrator::DoTearDown() { base_type::DoTearDown(); }
void TimeIntegrator::Run() {
    InitialCondition(GetTimeNow());
    CheckPoint();
    Dump();

    while (!Done()) {
        VERBOSE << " [ TIME :" << std::setw(5) << GetTimeNow() << "   ] ";
        Synchronize(0);
        NextStep();
        if (GetCheckPointInterval() > 0 && (GetStepNumber() % GetCheckPointInterval() == 0)) { CheckPoint(); };
        if (GetDumpInterval() > 0 && (GetStepNumber() % GetDumpInterval() == 0)) { Dump(); };
    }
}
void TimeIntegrator::NextStep() {
    Advance(GetTimeNow(), GetTimeStep());
    m_pimpl_->m_time_now_ += m_pimpl_->m_time_step_;
    base_type::NextStep();
}

void TimeIntegrator::Synchronize(int level) { base_type::Synchronize(level); }
bool TimeIntegrator::Done() const { return GetTimeNow() >= GetTimeEnd() || GetStepNumber() >= GetMaxStep(); }
}
}