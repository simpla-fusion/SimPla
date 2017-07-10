//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_CONTEXT_H
#define SIMPLA_CONTEXT_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/Serializable.h>
#include <map>
#include "Atlas.h"
#include "DomainBase.h"
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

/**
 * Context is a container of Model,Atlas,Domains
 */
class Context : public SPObject, public data::EnableCreateFromDataTable<Context> {
    SP_OBJECT_HEAD(Context, SPObject)
   public:
    explicit Context(const std::string &s_name = "");
    ~Context() override;

    SP_DEFAULT_CONSTRUCT(Context)
    DECLARE_REGISTER_NAME(Context)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    Atlas &GetAtlas() const;

    void SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &);
    std::shared_ptr<DomainBase> GetDomain(std::string const &k) const;

    std::map<std::string, std::shared_ptr<DomainBase>> &GetAllDomains();
    std::map<std::string, std::shared_ptr<DomainBase>> const &GetAllDomains() const;
    std::map<std::string, std::shared_ptr<AttributeDesc>> CollectRegisteredAttributes() const;

    void InitialCondition(Patch *patch, Real time_now);
    void BoundaryCondition(Patch *patch, Real time_now, Real time_dt);
    void Advance(Patch *patch, Real time_now, Real time_dt);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_CONTEXT_H
