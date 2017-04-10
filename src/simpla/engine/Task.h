//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_TASK_H
#define SIMPLA_TASK_H

#include <memory>
#include <set>
#include <vector>
#include "Attribute.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/concept/Printable.h"
#include "simpla/design_pattern/Observer.h"
#include "simpla/engine/SPObject.h"
#include "simpla/toolbox/Log.h"

namespace simpla {
namespace engine {

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
class Task : public concept::Configurable {
    SP_OBJECT_BASE(Task)
   public:
    Task(std::shared_ptr<data::DataTable> const &t = nullptr);
    Task(Task const &other);
    virtual ~Task();
    virtual void swap(Task &other);
    virtual void Register(AttributeGroup *);
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual void Initialize();
    virtual void Process(){};
    virtual bool Update();

    virtual void Run(Real dt);

    static bool RegisterCreator(std::string const &k, std::function<Task *()> const &);
    static Task *Create(std::string const &k);

    template <typename U>
    static bool RegisterCreator(std::string const &k) {
        return RegisterCreator(k, [&]() { return new U; });
    }
};

}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_TASK_H
