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
struct AttributeView;
struct Mesh;
struct Patch;

/**
 * @brief
 *
 *
 * @startuml
 * title Create/Destroy
 * actor  Main
 * create Worker
 * Main -> Worker : <<create>>
 * activate Worker
 *    create Mesh
 *    Worker -> Mesh : << create >>
 *    create Attribute
 *    Worker -> Attribute : << create >>
 *    activate Attribute
 *         Attribute -> Worker : request mesh()
 *         Worker --> Attribute : return Mesh*
 *        Attribute -> Worker : register   attr
 *    deactivate Attribute
 *    Worker --> Main : done
 * deactivate Worker
 *     ... ~~ DO sth. ~~ ...
 * Main -> Worker : << destroy >>
 * activate Worker
 *     Worker -> Attribute : << destroy >>
 *     activate Attribute
 *         Attribute -> Worker  : deregister
 *         Worker --> Attribute : done
 *         Attribute --> Worker : done
 *     deactivate Attribute
 *     Worker --> Main : done
 * deactivate Worker
 *
 *
 *    destroy Attribute
 *    destroy Mesh
 *    destroy Worker
 * @enduml
 *
 * @startuml
 * title Initialize/Finalize
 * actor  Main
 *
 * participant Worker as EMWorker << Generated >>
 * participant Worker as Worker << base >>
 * Main -> EMWorker : << Initialize >>
 * activate EMWorker
 * EMWorker -> Worker : << Initialize >>
 *     activate Worker
 *        alt if Patch == nullptr
 *             create Patch
 *             Worker -> Patch :<<create>>
 *        end
 *        Worker -> Patch          : mesh block
 *        Patch --> Worker         : return mesh block
 *        alt if Mesh == nullptr
 *         create Mesh
 *         Worker -> Mesh :<<create>>
 *        end
 *        Worker -> Mesh           : send mesh_block
 *        activate Mesh
 *             Mesh -> Mesh        : Deploy
 *             activate Mesh
 *             deactivate Mesh
 *             Mesh --> Worker     : done
 *        deactivate Mesh
 *        Worker -> Patch   : find DataBlock at MeshBlock
 *        activate Patch
 *             Patch --> Worker    : DataBlock
 *        deactivate Patch
 *        Worker -> Attribute      : send DataBlock & Mesh
 *        activate Attribute
 *             alt DataBlock == nullptr
 *                 Attribute-> Attribute : CreateDataBlock()
 *                 activate Attribute
 *                 deactivate Attribute
 *             end
 *             Attribute --> Worker : done
 *
 *        deactivate Attribute
 *        Worker --> EMWorker   : done
 *     deactivate Worker
 *     EMWorker->EMWorker: do ~~Initialize~~ things
 *     activate EMWorker
 *     deactivate EMWorker
 *      EMWorker->Main: done
 * deactivate EMWorker
 *
 *         ... ~~ DO sth. ~~ ...
 *     Main ->EMWorker: << Finalize >>
 * activate EMWorker
 *     EMWorker->EMWorker: do ~~Finalize~~ things
 *     activate EMWorker
 *     deactivate EMWorker
 *     EMWorker -> Worker : << Finalize >>
 *     activate Worker
 *         Worker->Worker: do ~~Finalize~~ thing
 *         Worker->Patch : push MeshBlock
 *             activate Patch
 *                 Patch-->Worker : done
 *             deactivate Patch
 *         Worker->Patch : push DataBlock
 *            activate Patch
 *                 Patch-->Worker : done
 *            deactivate Patch
 *         Worker -> Mesh : << Finalize >>
 *         activate Mesh
 *             Mesh -> Mesh : free MeshBlock
 *             Mesh --> Worker : done
 *         deactivate Mesh
 *         Worker -> Attribute : << Finalize >>
 *         activate Attribute
 *              Attribute --> Worker : done
 *        deactivate Attribute
 *         Worker--> EMWorker: done
 *
 *     deactivate Worker
 *     EMWorker--> Main: done
 * deactivate EMWorker
 * deactivate Main
 * @enduml
 */
class Worker : public Object, public concept::Configurable, public concept::Printable {
    SP_OBJECT_HEAD(Worker, Object)
   public:
    Worker();

    virtual ~Worker();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    virtual std::shared_ptr<Mesh> create_mesh() = 0;

    std::shared_ptr<Patch> patch() { return m_patch_; }
    virtual void Accept(std::shared_ptr<Patch> p = nullptr);
    virtual void Release();

    virtual void Initialize();
    virtual void PreProcess();
    virtual void PostProcess();
    virtual void Finalize();

    virtual void NextTimeStep(Real data_time, Real dt){};

    virtual void Sync();

    virtual void SetPhysicalBoundaryConditions(Real time){};

    virtual void Connect(AttributeView *attr) { m_attrs_.insert(attr); };

    virtual void Disconnect(AttributeView *attr) { m_attrs_.erase(attr); }

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
    std::set<AttributeView *> m_attrs_;
    std::unique_ptr<simpla::model::Model> m_model_;
};
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
