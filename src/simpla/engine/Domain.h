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
class Worker;
// class MeshBlock;
// class DataBlock;
// class Domain;
// class Task;
//
class Domain : public SPObject, public concept::Serializable<Domain> {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    Domain();
    Domain(const Domain &) = delete;
    Domain(Domain &&) = delete;
    ~Domain();

    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable> const &);

    void SetChart(std::shared_ptr<Chart> const &);
    std::shared_ptr<Chart> const &GetChart() const;

    void SetGeoObject(std::shared_ptr<geometry::GeoObject> const &geo_object) const;
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    void SetWorker(std::shared_ptr<Worker> const &);
    std::shared_ptr<Worker> const &GetWorker() const;

    void AddBoundaryCondition(std::shared_ptr<Worker> const &, std::shared_ptr<geometry::GeoObject> const &g = nullptr);

    //    void Register(AttributeGroup *);
    //    void Deregister(AttributeGroup *);

    void Update(Patch *, Real time_now = 0, Real time_dt = 0);

   private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
