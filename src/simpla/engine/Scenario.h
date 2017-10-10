//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_SCENARIO_H
#define SIMPLA_SCENARIO_H

#include "Attribute.h"
#include "EngineObject.h"
#include "simpla/geometry/GeoObject.h"
namespace simpla {
namespace engine {
class MeshBase;
class DomainBase;
class Atlas;

class Scenario : public EngineObject {
    SP_OBJECT_HEAD(Scenario, EngineObject)

    virtual void TagRefinementCells(Real time_now);

    virtual void Synchronize(int level);
    virtual void NextStep();
    virtual void Run();
    virtual bool Done() const;

    virtual void CheckPoint(size_type step_num) const;
    virtual void Dump() const;

    void DoInitialize() override;
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;
    void DoFinalize() override;

    virtual Real GetTime() const { return 0.0; }

    void SetStepNumber(size_type s);
    size_type GetStepNumber() const;

    std::shared_ptr<Atlas> GetAtlas() const;

    size_type SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d);
    template <typename U>
    size_type SetDomain(std::string const &k, std::shared_ptr<geometry::GeoObject> const &g) {
        static_assert(std::is_base_of<DomainBase, U>::value, "illegal domain type!");
        auto res = U::New();
        res->SetBoundary(g);
        return SetDomain(k, res);
    }

    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    std::map<std::string, std::shared_ptr<DomainBase>> &GetDomains();
    std::map<std::string, std::shared_ptr<DomainBase>> const &GetDomains() const;

    size_type DeletePatch(id_type);
    id_type SetPatch(id_type id, const std::shared_ptr<data::DataNode> &p);
    std::shared_ptr<data::DataNode> GetPatch(id_type id) const;
    std::shared_ptr<data::DataNode> GetPatch(id_type id);

    //    std::map<std::string, std::shared_ptr<data::DataNode>> const &GetAttributes() const;
    //    std::map<std::string, std::shared_ptr<data::DataNode>> &GetAttributes();

    std::shared_ptr<Attribute> GetAttribute(std::string const &key);
    std::shared_ptr<Attribute> GetAttribute(std::string const &key) const;
    template <typename... Args>
    size_type ConfigureAttribute(std::string const &name, Args &&... args) {
        size_type success = 0;
        if (auto attr = GetAttribute(name)) { success = attr->db()->SetValue(std::forward<Args>(args)...); }
        return success;
    }

    template <typename U>
    size_type ConfigureAttribute(std::string const &name, std::string const &key, U const &u) {
        size_type success = 0;
        if (auto attr = GetAttribute(name)) { success = attr->db()->SetValue<U>(key, u); }
        return success;
    }

    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> const &GetRange(std::string const &k) const;
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
