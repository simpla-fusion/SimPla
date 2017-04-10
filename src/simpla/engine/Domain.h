//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <memory>
#include "Attribute.h"
#include "Chart.h"
#include "simpla/mpl/macro.h"

namespace simpla {
namespace engine {
// class Attribute;
class Mesh;
class Patch;
// class MeshBlock;
// class DataBlock;
// class Domain;
// class Task;
//
class Domain : public SPObject {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    Domain(std::shared_ptr<data::DataTable> const &t = nullptr);
    Domain(const Domain &);
    Domain(Domain &&);
    ~Domain();

    void swap(Domain &);
    void Initialize();
    void Finalize();

    AttributeGroup const &GetAttributes() const;

    void SetChart(std::shared_ptr<Chart> const &);
    std::shared_ptr<Chart> const &GetChart() const;

    /**
     * @brief  Set geometric boudary
     * @param geo_object define the boundary of Domain
     * @param pos -1 domain is inside geo_object,
     *             0 domain is on the boundary
     *             1 domain is outside  geo_object,
     */
    enum { INSIDE = -1, BOUNDARY = 0, OUTSIDE = 1 };
    void SetGeoObject(std::shared_ptr<geometry::GeoObject> const & geo_object, int pos = INSIDE) const;
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    void SetWorker(std::shared_ptr<Worker> const &);
    std::shared_ptr<Worker> const &GetWorker() const;

    void Push(Patch);
    Patch Pop();

    bool isValid() const;

   private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
