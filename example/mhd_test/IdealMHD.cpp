//
// Created by salmon on 17-5-28.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/physics/PhysicalConstants.h>
namespace simpla {
using namespace algebra;
using namespace data;
using namespace engine;

template <typename TM>
class IdealMHD : public engine::Domain {
    SP_OBJECT_HEAD(IdealMHD<TM>, engine::Domain)

   public:
    DOMAIN_HEAD(IdealMHD, TM)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    void InitialCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real dt) override;
    void Advance(Real time_now, Real dt) override;

    DOMAIN_DECLARE_FIELD(ne, VOLUME, 1);
};

REGISTER_CREATOR_TEMPLATE(IdealMHD, mesh::RectMesh)

template <typename TM>
std::shared_ptr<data::DataTable> IdealMHD<TM>::Serialize() const {
    auto res = engine::Domain::Serialize();
    return res;
};
template <typename TM>
void IdealMHD<TM>::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    engine::Domain::Deserialize(cfg);
}

template <typename TM>
void IdealMHD<TM>::InitialCondition(Real time_now) {}
template <typename TM>
void IdealMHD<TM>::BoundaryCondition(Real time_now, Real dt) {}
template <typename TM>
void IdealMHD<TM>::Advance(Real time_now, Real dt) {}

}  // namespace simpla  {