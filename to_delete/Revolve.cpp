//
// Created by salmon on 17-8-16.
//
#include "Revolve.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataEntry> RevolveZ::Serialize() const {
    auto tdb = base_type::Serialize();

    tdb->template SetValue("Axis", m_phi_axe_);
    tdb->template SetValue("Origin", m_axis_.o);
    tdb->template SetValue("Phi", nTuple<Real, 2>{m_angle_min_, m_angle_max_});

    tdb->Set("2DShape", base_obj->Serialize());

    return tdb;
};
void RevolveZ::Deserialize(std::shared_ptr<data::DataEntry>const & cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla