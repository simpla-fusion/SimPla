//
// Created by salmon on 17-2-12.
//

#ifndef SIMPLA_DOMAINVIEW_H
#define SIMPLA_DOMAINVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/mpl/macro.h>
#include <memory>
#include <set>
#include "AttributeView.h"
namespace simpla {
namespace geometry {
class GeoObject;
}
namespace data {
class DataEntity;
class DataBlock;
}
namespace engine {
class Domain;
class Worker;
class MeshView;
class MeshBlock;
class AttributeView;
class Patch;

class DomainView : public concept::Configurable {
   public:
    DomainView(std::shared_ptr<data::DataEntity> const &p = nullptr,
               std::shared_ptr<geometry::GeoObject> const &g = nullptr);
    virtual ~DomainView();
    virtual void Initialize();
    virtual void Finalize();

    virtual void SetMesh(MeshView const *) = delete;
    virtual MeshView const *GetMesh() const;
    std::shared_ptr<geometry::GeoObject> GetGeoObject() const;
    virtual void PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataEntity> const &);
    virtual void PushData(std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataEntity>> const &);
    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataEntity>> PopData();

    virtual void Run(Real dt);

    std::pair<std::shared_ptr<Worker>, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename>
struct DomainViewAdapter : public DomainView {};
}  // namespace engine {
}  // namespace simpla {

#endif  // SIMPLA_DOMAINVIEW_H
