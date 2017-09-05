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
    virtual void Dump();
    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    template <typename TM, typename... Args>
    std::shared_ptr<TM> SetMesh(Args &&... args) {
        auto p = TM::New(std::forward<Args>(args)...);
        SetMesh(p);
        return p;
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
    template <typename U>
    std::shared_ptr<U> GetModelAs(std::string const &k) const {
        return std::dynamic_pointer_cast<U>(GetModel(k));
    }

    std::shared_ptr<DomainBase> SetDomain(std::string const &k, std::shared_ptr<DomainBase> d);
    template <typename U, typename... Args>
    std::shared_ptr<U> SetDomain(std::string const &k, Args &&... args) {
        static_assert(std::is_base_of<DomainBase, U>::value, " Illegal domain type!");
        auto res = U::New(GetMesh(), GetModel(k), std::forward<Args>(args)...);
        SetDomain(k, res);
        return res;
    };
    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    template <typename U>
    std::shared_ptr<U> GetDomainAs(std::string const &k) const {
        return std::dynamic_pointer_cast<U>(GetDomain(k));
    }
    std::map<std::string, std::shared_ptr<DomainBase>> &GetDomains();
    std::map<std::string, std::shared_ptr<DomainBase>> const &GetDomains() const;

    std::shared_ptr<data::DataNode> Pop() override;
    int Push(std::shared_ptr<data::DataNode> const &p) override;

   private:
    void SetMesh(std::shared_ptr<MeshBase> const &);
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_SCENARIO_H
