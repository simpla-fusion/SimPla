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

namespace simpla {
namespace model {
class Model;
}
namespace mesh {
struct MeshBlock;
struct DataBlock;
struct MeshView;
}
namespace engine {
struct Patch;
struct AttributeView;
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
 *
 *         ... ~~ DO sth. ~~ ...
 * Main ->Worker: << Finalize >>
 * activate Worker
 *     Worker->EMWorker:  << Finalize >>
 *     activate EMWorker
 *        EMWorker -> EMWorker : Finalize
 *        EMWorker -->Worker:  done
 *     deactivate EMWorker
 *
 *     Worker -> AttributeView : << Finalize >>
 *     activate AttributeView
 *          AttributeView  -> Patch  : push DataBlock at MeshBlockId
 *          Patch --> AttributeView  : Done
 *          AttributeView --> Worker : done
 *     deactivate AttributeView
 *     Worker--> Main: done
 * deactivate Worker
 * deactivate Main
 * @enduml
 *
   @startuml
      (*)--> Accept(Patch)
      if "Mesh is defined" then
        --> [true] Initialize
      else
        --> [false] CreateMesh
        --> Push Mesh to Patch
      endif
      --> "Initialize all attributes" as Initialize
      --> PreProcess
      --> Process
      --> [ !isDone ] Process
      --> [ isDone ] PostProcess
      --> Finalize
      --> Release
      --> (*)
   @enduml
 */
class Worker : public Object, public concept::Configurable, public concept::Printable {
    SP_OBJECT_HEAD(Worker, Object)
   public:
    Worker();
    virtual ~Worker();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual std::shared_ptr<MeshView> create_mesh() = 0;
    std::shared_ptr<Patch> patch() { return m_patch_; }
    virtual void Accept(std::shared_ptr<Patch> p = nullptr);

    virtual void Initialize();
    virtual void PreProcess();
    virtual void NextTimeStep(Real data_time, Real dt){};
    virtual void Sync();
    //    virtual void Process() = 0;
    virtual void PostProcess();
    virtual void Finalize();
    virtual void Release();

    virtual void SetPhysicalBoundaryConditions(Real time){};


    template <typename TFun>
    void ForeachAttr(TFun const &fun) {
        for (auto attr : m_attrs_) { fun(attr); }
    }
    virtual simpla::model::Model *model() = 0;
    virtual simpla::model::Model const *model() const = 0;
    virtual MeshView *mesh() = 0;
    virtual MeshView const *mesh() const = 0;

    virtual void Connect(AttributeView *attr) { m_attrs_.insert(attr); };
    virtual void Disconnect(AttributeView *attr) { m_attrs_.erase(attr); }
   private:
    std::shared_ptr<Patch> m_patch_;
    std::set<AttributeView *> m_attrs_;
};
}  // namespace engine {

}  // namespace simpla
#endif  // SIMPLA_WORKER_H
