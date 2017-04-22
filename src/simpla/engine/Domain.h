//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include "Attribute.h"
#include "Chart.h"
#include "simpla/geometry/GeoObject.h"
#include "simpla/utilities/macro.h"
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
class Domain : public SPObject, public data::Serializable {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    Domain();
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain)

    void Register(AttributeGroup *);
    void Deregister(AttributeGroup *);

    std::shared_ptr<Patch> Pop();
    void Push(std::shared_ptr<Patch> p);

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    void SetChart(std::shared_ptr<Chart>);
    std::shared_ptr<Chart> GetChart() const;

    void SetGeoObject(std::shared_ptr<geometry::GeoObject> geo_object);
    std::shared_ptr<geometry::GeoObject> GetGeoObject() const;

    void SetWorker(std::shared_ptr<Worker>);
    std::shared_ptr<Worker> GetWorker() const;

    void AddBoundaryCondition(std::shared_ptr<Worker>, std::shared_ptr<geometry::GeoObject> g = nullptr);

    void InitializeCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);
    void Advance(Real time_now, Real time_dt);

    void Initialize() override;
    void SetUp() override;
    void TearDown() override;
    void Finalize() override;

    virtual void InitializeData(Real time_now);

   private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
