//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <memory>
#include "Attribute.h"
#include "simpla/geometry/GeoObject.h"
#include "simpla/mpl/macro.h"

namespace simpla {
namespace engine {
class Attribute;
class Mesh;
class MeshBlock;
class DataBlock;
class Domain;
class Worker;

class Domain : public concept::Configurable, public std::enable_shared_from_this<Domain> {
   public:
    Domain(std::shared_ptr<data::DataTable> const &m, std::shared_ptr<geometry::GeoObject> const &g = nullptr);
    Domain(std::shared_ptr<Mesh> const &m);
    Domain(const Domain &);
    virtual ~Domain();
    virtual void Initialize();
    virtual void Finalize();
    std::shared_ptr<Domain> Clone() const;
    void SetMeshView(std::shared_ptr<Mesh> const &);
    std::shared_ptr<Mesh> const &GetMeshView() const;
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    virtual void PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &);
    virtual void PushData(std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> const &);
    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> PopData();

    virtual void Run(Real dt);

    std::pair<std::shared_ptr<Worker>, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
