//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <simpla/design_pattern/Observer.h>
#include <simpla/toolbox/Log.h>
#include <memory>
#include <set>
#include <vector>

#include <simpla/concept/Configurable.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>

#include <simpla/model/Model.h>
#include "MeshCommon.h"

namespace simpla {
namespace model {
class Model;
}
namespace mesh {
struct MeshBlock;
struct DataBlock;
struct Attribute;
struct Mesh;
struct Patch;

class Worker : public Object, public concept::Printable, public concept::Serializable, public concept::Configurable {
    SP_OBJECT_HEAD(Worker, Object)
   public:
    Worker();

    virtual ~Worker();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    virtual std::shared_ptr<Mesh> create_mesh() = 0;

    virtual void Load(data::DataTable const &) { UNIMPLEMENTED; }

    virtual void Save(data::DataTable *) const { UNIMPLEMENTED; }

    std::shared_ptr<Patch> patch() { return m_patch_; }
    virtual void Accept(std::shared_ptr<Patch> p = nullptr);
    virtual void Release();

    virtual void Deploy();
    virtual void Initialize();
    virtual void PreProcess();
    virtual void PostProcess();
    virtual void Finalize();
    virtual void Destroy();

    virtual void NextTimeStep(Real data_time, Real dt){};

    virtual void Sync();

    virtual void SetPhysicalBoundaryConditions(Real time){};

    virtual void Connect(Attribute *attr) { m_attrs_.insert(attr); };

    virtual void Disconnect(Attribute *attr) { m_attrs_.erase(attr); }

    template <typename TFun>
    void ForeachAttr(TFun const &fun) {
        for (auto attr : m_attrs_) { fun(attr); }
    }
    simpla::model::Model &model() {
        ASSERT(m_model_ != nullptr);
        return *m_model_;
    }
    simpla::model::Model const &model() const {
        ASSERT(m_model_ != nullptr);
        return *m_model_;
    }
    virtual Mesh *mesh() { return m_mesh_.get(); };
    virtual Mesh const *mesh() const { return m_mesh_.get(); };

    std::shared_ptr<Patch> m_patch_;
    std::shared_ptr<Mesh> m_mesh_;
    std::set<Attribute *> m_attrs_;
    std::unique_ptr<simpla::model::Model> m_model_;
};
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
