//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_CONTEXT_H
#define SIMPLA_CONTEXT_H

#include <simpla/SIMPLA_config.h>
#include <map>
#include "Atlas.h"
#include "Model.h"
#include "Patch.h"
#include "Worker.h"

namespace simpla {
namespace engine {

/**
 * @brief
 *
 * @startuml
 * start
 * repeat
 *  : DomainView::Dispatch(Domain & d);
 *  : DomainView::Update() ;
 *  : DomainView:: ;
 * repeat while (more Domain?)
 *
 * stop
 * @enduml
 *
 * @startuml
 *  Context -> DomainView: Dispatch(Domain &)
 *  DomainView -->Context : Done
 *  Context -> DomainView: Update()
 *  activate DomainView
 *      DomainView -> DomainView :Update
 *      activate DomainView
 *          DomainView -> MeshView: Dispatch(Domain::mesh_block)
 *          MeshView --> DomainView: Done
 *          DomainView -> MeshView: Update()
 *          activate MeshView
 *              MeshView -> MeshView : Update
 *              MeshView --> DomainView : Done
 *          deactivate MeshView
 *          DomainView -> Worker  : Update()
 *          activate Worker
 *              activate Worker
 *                    Worker -> AttributeView : Update
 *                    activate AttributeView
 *                          AttributeView -> DomainView : require DataBlock
 *                          DomainView --> AttributeView: return DataBlock
 *                          AttributeView --> Worker : Done
 *                    deactivate AttributeView
 *              deactivate Worker
 *              Worker --> DomainView : Done
 *          deactivate Worker
 *      deactivate DomainView
 *      DomainView --> Context: Done
 *  deactivate DomainView
 * @enduml
 */
class Context : public concept::Configurable {
    SP_OBJECT_BASE(Context)

   public:
    Context(std::shared_ptr<data::DataTable> const &t = nullptr);
    virtual ~Context();

    virtual void Initialize();
    virtual void Finalize();
    bool IsInitialized() const;

    virtual void Advance(Real dt, int level = 0);
    virtual void Synchronize(int from_level = 0, int to_level = 0);

    Atlas &GetAtlas() const;
    Model &GetModel() const;

    std::shared_ptr<data::DataTable> GetPatches() const;

    bool SetWorker(std::string const &d_name, std::shared_ptr<Worker> const &p);
    void RemoveWorker(std::string const &d_name);
    std::shared_ptr<Worker> GetWorker(std::string const &d_name) const;

    bool RegisterAttribute(std::string const &key, std::shared_ptr<Attribute> const &);
    template <typename TV, int IFORM = VERTEX, int DOF = 1>
    bool RegisterAttribute(std::string const &key) {
        return RegisterAttribute(key, std::make_shared<AttributeDesc<TV, IFORM, DOF>>());
    };

    void DeregisterAttribute(std::string const &key);
    std::shared_ptr<Attribute> const &GetAttribute(std::string const &key) const;
    std::map<std::string, std::shared_ptr<Attribute>> const &GetAllAttributes() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_CONTEXT_H
