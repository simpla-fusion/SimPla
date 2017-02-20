//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_MANAGER_H
#define SIMPLA_MANAGER_H

#include <simpla/SIMPLA_config.h>
#include <simpla/model/Model.h>
#include <map>
#include "Atlas.h"
#include "DomainView.h"
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
 *  Manager -> DomainView: Dispatch(Domain &)
 *  DomainView -->Manager : Done
 *  Manager -> DomainView: Update()
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
 *      DomainView --> Manager: Done
 *  deactivate DomainView
 * @enduml
 */
class Manager : public concept::Printable, public concept::Configurable {
   public:
    Manager();
    virtual ~Manager();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return os; }
    model::Model const &GetModel() const;
    model::Model &GetModel();
    void SetDomainView(std::string const &domain_type_name, std::shared_ptr<DomainView> const &p = nullptr);
    void SetDomainView(id_type domain_type_id, std::shared_ptr<DomainView> const &p = nullptr);
    DomainView const &GetDomainView(id_type id) const;
    DomainView const &GetDomainView(std::string const &d_name) const;
    DomainView &GetDomainView(id_type id);
    DomainView &GetDomainView(std::string const &d_name);
    void Update();
    void Evaluate();

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename U>
struct ManagerAdapter : public Manager, public U {};
}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_MANAGER_H
