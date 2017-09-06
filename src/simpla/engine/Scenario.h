//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_SCENARIO_H
#define SIMPLA_SCENARIO_H

#include "Attribute.h"
#include "EngineObject.h"

namespace simpla {
namespace engine {
class MeshBase;
class DomainBase;
class Model;
class Atlas;
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

    std::shared_ptr<Atlas> GetAtlas() const;
    std::shared_ptr<Model> GetModel() const;

    size_type SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d);
    template <typename U>
    size_type SetDomain(std::string const &k) {
        static_assert(std::is_base_of<DomainBase, U>::value, "illegal domain type!");
        return SetDomain(k, U::New());
    }

    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    std::map<std::string, std::shared_ptr<DomainBase>> &GetDomains();
    std::map<std::string, std::shared_ptr<DomainBase>> const &GetDomains() const;

    size_type DeletePatch(id_type);
    id_type SetPatch(id_type id, const std::shared_ptr<data::DataNode> &p);
    std::shared_ptr<data::DataNode> GetPatch(id_type id) const;
    std::shared_ptr<data::DataNode> GetPatch(id_type id);

    std::map<std::string, std::shared_ptr<Attribute>> &GetAttributes() const;

    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> const &GetRange(std::string const &k) const;
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
