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
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/engine/Object.h>

#include <simpla/model/Model.h>

namespace simpla {
namespace engine {

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
class Worker : public AttributeViewBundle, public concept::Configurable, public concept::Printable {
    SP_OBJECT_BASE(Worker)

   public:
    Worker();
    virtual ~Worker();
    std::shared_ptr<Worker> &next();
    std::shared_ptr<Worker> const &Worker::next() const;

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual void Initialize() = 0;
    virtual void Process() = 0;

    using AttibuteViewBundle::isUpdate;

    void Update();
    void Evaluate();


   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
