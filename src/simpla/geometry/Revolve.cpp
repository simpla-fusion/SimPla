//
// Created by salmon on 17-8-16.
//
#include "Revolve.h"
namespace simpla {
namespace geometry {

void RevolveZ::Serialize(std::shared_ptr<data::DataEntity> const &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        tdb->template SetValue("Axis", m_phi_axe_);
        tdb->template SetValue("Origin", m_origin_);
        tdb->template SetValue("Phi", nTuple<Real, 2>{m_angle_min_, m_angle_max_});

        base_obj->Serialize(tdb->Get("2DShape"));
    }
};
void RevolveZ::Deserialize(std::shared_ptr<const data::DataEntity> const &cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla