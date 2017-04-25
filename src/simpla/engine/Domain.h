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
class MeshBase;
class Patch;
class Worker;
class Model;
// class MeshBlock;
// class DataBlock;
// class Domain;
// class Task;

class Domain : public SPObject, public data::Serializable {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    Domain();
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain)

    void Register(AttributeGroup *);
    void Deregister(AttributeGroup *);

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    void SetChart(std::shared_ptr<Chart>);
    std::shared_ptr<Chart> GetChart() const;
    template <typename C>
    void SetChart() {
        SetChart(std::make_shared<C>());
    }

    void SetWorker(std::shared_ptr<geometry::GeoObject> g, std::shared_ptr<Worker> w);
    template <typename U>
    void SetWorker(std::shared_ptr<geometry::GeoObject> g, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        SetWorker(g, std::dynamic_pointer_cast<Worker>(std::make_shared<U>(GetChart(), g)));
    };

    void RegisterModel(Model *);

    void Initialize() override;
    void SetUp() override;
    void TearDown() override;
    void Finalize() override;

    void InitializeData(Patch *p, Real time_now);
    void InitializeCondition(Patch *p, Real time_now);
    void BoundaryCondition(Patch *p, Real time_now, Real time_dt);
    void Advance(Patch *p, Real time_now, Real time_dt);

   private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
