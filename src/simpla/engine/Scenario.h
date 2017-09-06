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
    virtual void TagRefinementCells(Real time_now);

    virtual void Synchronize();
    virtual void NextStep();
    virtual void Run();
    virtual bool Done() const;

    virtual void Dump() const;

    void DoInitialize() override;
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;
    void DoFinalize() override;

    template <typename TM, typename... Args>
    std::shared_ptr<TM> SetMesh(Args &&... args) {
        auto p = TM::New(std::forward<Args>(args)...);
        SetMesh(p);
        return p;
    };
    std::shared_ptr<MeshBase> GetMesh() const;

    std::shared_ptr<Atlas> GetAtlas() const;

    size_type SetModel(std::shared_ptr<Model> const &m);
    std::shared_ptr<Model> GetModel(std::string const &k = "") const;
    template <typename U>
    std::shared_ptr<U> GetModelAs(std::string const &k = "") const {
        return std::dynamic_pointer_cast<U>(GetModel(k));
    }

    size_type SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d);
    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    std::map<std::string, std::shared_ptr<DomainBase>> &GetDomains();
    std::map<std::string, std::shared_ptr<DomainBase>> const &GetDomains() const;

    size_type DeletePatch(id_type);
    id_type SetPatch(id_type id, const std::shared_ptr<data::DataNode> &p);
    std::shared_ptr<data::DataNode> GetPatch(id_type id) const;
    std::shared_ptr<data::DataNode> GetPatch(id_type id);

    //    std::shared_ptr<data::DataNode> Pop() override;
    //    int Push(std::shared_ptr<data::DataNode> const &p) override;

   private:
    void SetMesh(std::shared_ptr<MeshBase> const &);
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
