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
// class Attribute;
// class Mesh;
// class MeshBlock;
// class DataBlock;
// class Domain;
// class Task;
//
class Domain : public concept::Configurable {
   public:
    Domain(std::shared_ptr<geometry::GeoObject> const &g = nullptr,
           std::shared_ptr<data::DataTable> const &t = nullptr);
    Domain(const Domain &);
    Domain(Domain &&);
    ~Domain();

    void swap(Domain &);
    void Initialize();
    void Finalize();

    AttributeBundle const &GetAttributes() const;

    void SetMeshView(std::shared_ptr<Mesh> const &);
    std::shared_ptr<Mesh> const &GetMeshView() const;

    void SetWorker(std::shared_ptr<Worker> const &);
    std::shared_ptr<Worker> const &GetWorker() const;

    void SetGeoObject(std::shared_ptr<geometry::GeoObject> const &) const;
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    void Push(std::shared_ptr<Patch> const &);
    std::shared_ptr<Patch> Pop();

   private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
