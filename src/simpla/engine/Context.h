//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_CONTEXT_H
#define SIMPLA_CONTEXT_H

#include <simpla/SIMPLA_config.h>
#include <map>
#include "Atlas.h"
#include "DomainView.h"
#include "Model.h"
#include "Patch.h"

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
    virtual void Advance(Real dt, int level = 0);
    virtual void Synchronize(int from_level = 0, int to_level = 0);
    Atlas &GetAtlas() const;
    Model &GetModel() const;
    std::shared_ptr<data::DataTable> GetPatches() const;
    void SetDomain(std::string const &d_name, std::shared_ptr<Domain> const &p);
    std::shared_ptr<Domain> GetDomain(std::string const &d_name) const;
    std::map<std::string, std::shared_ptr<Domain>> const &GetAllDomains() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_CONTEXT_H
