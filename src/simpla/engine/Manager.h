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
class Manager : public SPObject, public concept::Printable {
   public:
    Manager();
    virtual ~Manager();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    Atlas &GetAtlas() const;
    model::Model &GetModel() const;
    DomainView &GetDomainView(std::string const &d_name) const;
    std::shared_ptr<data::DataTable> GetAttributeDatabase() const;

    std::shared_ptr<DomainView> SetDomainView(std::string const &d_name, std::shared_ptr<data::DataEntity> const &p);

    std::shared_ptr<DomainView> SetDomainView(std::string const &d_name, std::shared_ptr<DomainView> const &p = nullptr,
                                              bool overwrite = false);

    static bool RegisterMeshCreator(std::string const &k, std::function<std::shared_ptr<MeshView>()> const &);
    static bool RegisterWorkerCreator(std::string const &k, std::function<std::shared_ptr<Worker>()> const &);

    void Initialize();
    bool Update();
    Real GetTime() const;
    void Run(Real dt);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_MANAGER_H
