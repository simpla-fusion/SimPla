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
class Model;

class Scenario : public EngineObject {
    SP_OBJECT_HEAD(Scenario, EngineObject)

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    template <typename TD, typename... Args>
    std::shared_ptr<MeshBase> SetMesh(Args &&... args) {
        SetMesh(TD::New(std::forward<Args>(args)...));
        return GetMesh();
    };
    std::shared_ptr<MeshBase> GetMesh() const;
    std::shared_ptr<Atlas> GetAtlas() const;

    std::shared_ptr<Model> AddModel(std::string const &k, std::shared_ptr<Model> m);
    template <typename U, typename... Args>
    std::shared_ptr<U> AddModel(std::string const &k, Args &&... args) {
        auto res = U::New(std::forward<Args>(args)...);
        AddModel(k, res);
        return res;
    };
    std::shared_ptr<Model> GetModel(std::string const &k) const;

    std::shared_ptr<DomainBase> SetDomain(std::string const &k, std::shared_ptr<DomainBase> d);
    template <typename U, typename... Args>
    std::shared_ptr<U> SetDomain(std::string const &k, Args &&... args) {
        static_assert(std::is_base_of<DomainBase, U>::value, " Illegal domain type!");
        auto res = U::New(GetMesh(), GetModel(k), std::forward<Args>(args)...);
        SetDomain(k, res);
        return res;
    };
    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    void Pop(std::shared_ptr<Patch> &p);
    void Push(std::shared_ptr<Patch> &p);

    virtual void InitialCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void ComputeFluxes(Real time_now, Real time_dt);
    virtual Real ComputeStableDtOnPatch(Real time_now, Real time_dt);
    virtual Real Advance(Real time_now, Real dt);
    virtual void TagRefinementCells(Real time_now);

    virtual void Run();

    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    SP_OBJECT_PROPERTY(Real, TimeNow);
    SP_OBJECT_PROPERTY(Real, TimeEnd);
    SP_OBJECT_PROPERTY(Real, TimeStep);
    SP_OBJECT_PROPERTY(Real, CFL);

   private:
    void SetMesh(std::shared_ptr<MeshBase> const &);
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
