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

#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObject.h>
#include "AttributeView.h"

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
class Worker : public SPObject, public AttributeViewBundle {
    SP_OBJECT_HEAD(Worker, SPObject)
   public:
    Worker(std::shared_ptr<MeshView> const &p = nullptr, std::shared_ptr<data::DataEntity> const &t = nullptr);
    virtual ~Worker();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual void Initialize();
    virtual void Process(){};
    virtual bool Update();

    //    virtual void SetMesh(MeshView const *);
    //    virtual MeshView const *GetMesh() const;
    //    virtual void PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataEntity> const &);
    //    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataEntity>> PopData();

    using AttributeViewBundle::SetMesh;
    using AttributeViewBundle::GetMesh;
    using AttributeViewBundle::PopData;
    using AttributeViewBundle::PushData;

    virtual void Run(Real dt);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

struct WorkerFactory {
   public:
    WorkerFactory();
    ~WorkerFactory();
    bool RegisterCreator(std::string const &k,
                         std::function<std::shared_ptr<Worker>(std::shared_ptr<MeshView> const &,
                                                               std::shared_ptr<data::DataTable> const &)> const &);
    template <typename U>
    bool RegisterCreator(std::string const &k) {
        RegisterCreator(k, [&](std::shared_ptr<MeshView> const &m, std::shared_ptr<data::DataTable> const &t) {
            return std::make_shared<U>(m, t);
        });
    }

    std::shared_ptr<Worker> Create(std::shared_ptr<MeshView> const &m, std::shared_ptr<data::DataEntity> const &p);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define GLOBAL_WORKER_FACTORY SingletonHolder<WorkerFactory>::instance()

template <typename U>
struct WorkerAdapter : public Worker, public U {
    WorkerAdapter() {}
    virtual ~WorkerAdapter(){};
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const {
        U::Print(os, indent);
        return os;
    }
    virtual void Initialize() { U::Initialize(); };
    virtual void Process() { U::Process(); };
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
