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

/**
 * @brief
 *
 *
@startuml
title Create/Destroy __Worker__
actor  Main
create Worker
Main -> Worker : <<create>>
activate Worker
   create Mesh
   Worker -> Mesh : << create >>
   create Attribute
   Worker -> Attribute : << create >>
   activate Attribute
        Attribute -> Worker : request mesh()
        Worker --> Attribute : return Mesh*
       Attribute -> Worker : register   attr
   deactivate Attribute
deactivate Worker
    ... ~~ DO sth. ~~ ...
Main -> Worker : << destroy >>
activate Worker
    Worker -> Attribute : << destroy >>
    activate Attribute
        Attribute -> Worker  : deregister
        Worker --> Attribute : done
        Attribute --> Worker : done
    deactivate Attribute
    Worker --> Main : done
deactivate Worker


   destroy Attribute
   destroy Mesh
   destroy Worker
@enduml

@startuml
title Deploy/Destroy  __Worker__
Main -> Worker : << deploy >>
activate Worker
   alt if Patch == nullptr
        create Patch
        Worker -> Patch :<<create>>
   end
   Worker -> Patch          : mesh block
   Patch --> Worker         : return mesh block
   Worker -> Mesh           : send mesh_block
   activate Mesh
        Mesh -> Mesh        : Deploy
        activate Mesh
        deactivate Mesh
        Mesh --> Worker     : done
   deactivate Mesh
   Worker -> Patch   : find DataBlock at MeshBlock
   activate Patch
        Patch --> Worker    : DataBlock
   deactivate Patch
   Worker -> Attribute      : send DataBlock
   activate Attribute
        alt DataBlock == nullptr
            Attribute -> Mesh : request MeshBlock
            Mesh --> Attribute : MeshBlock
            Attribute-> Attribute : create DataBlock
            activate Attribute
            deactivate Attribute
        end
        Attribute->Attribute : assign data block
        Attribute --> Worker : done
   deactivate Attribute

   Worker --> Main   : done
deactivate Worker

    ... ~~ DO sth. ~~ ...
Main -> Worker : << Destroy >>
activate Worker
   Worker -> Attribute : << Destroy >>
   activate Attribute
     Attribute -> Attribute : free DataBlock
     Attribute --> Worker : done
   deactivate Attribute
   Worker -> Mesh : << Destroy >>
   activate Mesh
     Mesh -> Mesh : free MeshBlock
     Mesh --> Worker : done
   deactivate Attribute
   Worker--> Main: done
deactivate Worker

deactivate Main
@enduml
 */
class Worker : public Object, public concept::Configurable, public concept::Printable, public concept::Serializable {
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
