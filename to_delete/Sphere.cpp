//
// Created by salmon on 17-7-21.
//

#include "Sphere.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Sphere)
Sphere::Sphere() {}
Sphere::~Sphere() {}
std::shared_ptr<data::DataEntry> Sphere::Serialize() const {
    auto tdb = base_type::Serialize();
    tdb->SetValue("Origin", m_axis_.o);
    tdb->SetValue("Radius", m_radius_);
    return tdb;
};
void Sphere::Deserialize(std::shared_ptr<data::DataEntry>const & cfg) {
    base_type::Deserialize(cfg);
    m_axis_.o = db()->GetValue("Origin", m_axis_.o);
    m_radius_ = db()->GetValue("Radius", m_radius_);
}
}
}