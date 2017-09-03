//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_SCENARIO_H
#define SIMPLA_SCENARIO_H

#include "Atlas.h"
#include "EngineObject.h"

namespace simpla {
namespace engine {
class MeshBase;
class DomainBase;
class Schedule;
class Model;

class Scenario : public EngineObject {
    SP_OBJECT_HEAD(Scenario, EngineObject)

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void SetAtlas(std::shared_ptr<Atlas> const &);
    std::shared_ptr<Atlas> GetAtlas() const;

    void SetMesh(std::shared_ptr<MeshBase> const &);
    std::shared_ptr<MeshBase> GetMesh() const;

    void SetModel(std::string const &k, std::shared_ptr<Model> const &);
    std::shared_ptr<const Model> GetModel(std::string const &k) const;

    //    template <typename TD>
    //    std::shared_ptr<TD> NewDomain(std::string const &k, std::shared_ptr<Model> const &m = nullptr);
    std::shared_ptr<DomainBase> NewDomain(std::string const &);
    std::shared_ptr<DomainBase> NewDomain(std::shared_ptr<const data::DataNode>);
    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    template <typename TD>
    std::shared_ptr<TD> NewSchedule();
    std::shared_ptr<Schedule> NewSchedule(std::shared_ptr<const data::DataNode>);
    std::shared_ptr<Schedule> GetSchedule() const;

    void Pop(const std::shared_ptr<Patch> &p);
    void Push(const std::shared_ptr<Patch> &p);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void ComputeFluxes(Real time_now, Real time_dt);
    Real ComputeStableDtOnPatch(Real time_now, Real time_dt);
    void Advance(Real time_now, Real dt);
    void TagRefinementCells(Real time_now);

    void Run();

   private:
    void SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &);
    void SetSchedule(std::shared_ptr<Schedule> const &);
};

std::shared_ptr<DomainBase> Scenario::NewDomain(std::string const &k) {
    //    if (m != nullptr) { SetModel(k, m); }
    //    SetDomain(k, TD::New(GetMesh(), GetModel(k)));
    return GetDomain(k);
};
std::shared_ptr<DomainBase> Scenario::NewDomain(std::shared_ptr<const data::DataNode> db) { return nullptr; }

template <typename TD>
std::shared_ptr<TD> Scenario::NewSchedule() {
    SetSchedule(TD::New(shared_from_this()));
    return GetSchedule();
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
