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
class Domain : public SPObject, public data::Serializable {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    Domain();
    Domain(std::shared_ptr<data::DataTable>);
    Domain(const Domain &) = delete;
    Domain(Domain &&) = delete;
    ~Domain();

    void Register(AttributeGroup *attr_grp);
    void Deregister(AttributeGroup *);


    std::shared_ptr<data::DataTable> Serialize() const;
    void Deserialize(std::shared_ptr<data::DataTable>);

//    void SetChart(std::string const &w);
//    void SetChart(std::shared_ptr<data::DataTable> const &w);
    void SetChart(std::shared_ptr<Chart>);
    std::shared_ptr<Chart> GetChart() const;

    void SetGeoObject(std::shared_ptr<geometry::GeoObject> geo_object);
    std::shared_ptr<geometry::GeoObject> GetGeoObject() const;

    void SetWorker(std::shared_ptr<Worker>);
    template <typename T>
    void SetWorker(T const &t) {
        return SetWorker(CreateWorker(t));
    }
    std::shared_ptr<Worker> const &GetWorker() const;

    std::shared_ptr<Worker> CreateWorker(std::string const &w) const;
    std::shared_ptr<Worker> CreateWorker(std::shared_ptr<data::DataTable> w) const;

    void AddBoundaryCondition(std::shared_ptr<Worker>, std::shared_ptr<geometry::GeoObject> g = nullptr);
    template <typename T>
    void AddBoundaryCondition(T const &t, std::shared_ptr<geometry::GeoObject> g = nullptr) {
        AddBoundaryCondition(CreateWorker(t), g);
    };

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
