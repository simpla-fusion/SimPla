//
// Created by salmon on 17-7-21.
//

#include "Sphere.h"
namespace simpla {
namespace geometry {
REGISTER_CREATOR(Sphere, Sphere)
Sphere::Sphere() {}
Sphere::~Sphere() {}
void Sphere::Serialize(std::shared_ptr<data::DataEntity> const &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        tdb->SetValue("Origin", m_origin_);
        tdb->SetValue("Radius", m_radius_);
    }
};
void Sphere::Deserialize(std::shared_ptr<const data::DataEntity> const &cfg) {
    base_type::Deserialize(cfg);
    m_origin_ = db().GetValue("Origin", m_origin_);
    m_radius_ = db().GetValue("Radius", m_radius_);
}
}
}