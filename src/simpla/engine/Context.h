//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_CONTEXT_H
#define SIMPLA_CONTEXT_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/Serializable.h>
#include <map>
#include "Atlas.h"
#include "Domain.h"
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
    explicit Context(std::string const &s_name = "");
    ~Context() override;

    SP_DEFAULT_CONSTRUCT(Context)
    DECLARE_REGISTER_NAME("Context")

    std::shared_ptr<DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &cfg) override;

    void Initialize() override;
    void Finalize() override;
    void Update() override;
    void TearDown() override;

    Atlas &GetAtlas() const;

    int GetNDims() const;
    box_type GetBoundBox() const;

    void SetChart(std::string const &s_name, std::shared_ptr<geometry::Chart> const &m);
    void SetChart(std::string const &s_name, std::shared_ptr<data::DataEntity> const &);
    std::shared_ptr<geometry::Chart> GetChart(std::shared_ptr<data::DataEntity> const &) const;
    std::shared_ptr<geometry::Chart> GetChart(std::string const &s_name = "Default") const;

    void SetGeoObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &m);
    void SetGeoObject(std::string const &k, std::shared_ptr<data::DataEntity> const &);
    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::shared_ptr<data::DataEntity> const &) const;
    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::string const &k) const;

    void SetMesh(std::string const &k, std::shared_ptr<MeshBase> const &m);
    void SetMesh(std::string const &k, std::shared_ptr<data::DataEntity> const &);
    std::shared_ptr<MeshBase> GetMesh(std::shared_ptr<data::DataEntity> const &) const;
    std::shared_ptr<MeshBase> GetMesh(std::string const &k = "Default") const;

    void SetDomain(std::string const &k, std::shared_ptr<Domain> const &);
    void SetDomain(std::string const &k, std::shared_ptr<data::DataEntity> const &);
    std::shared_ptr<Domain> GetDomain(std::shared_ptr<data::DataEntity> const &) const;
    std::shared_ptr<Domain> GetDomain(std::string const &k) const;

    std::map<std::string, std::shared_ptr<Domain>> &GetAllDomains();
    std::map<std::string, std::shared_ptr<Domain>> const &GetAllDomains() const;
    std::map<std::string, std::shared_ptr<AttributeDesc>> const &GetRegisteredAttribute() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_CONTEXT_H
