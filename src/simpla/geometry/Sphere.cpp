//
// Created by salmon on 17-7-21.
//

#include "Sphere.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Sphere)
Sphere::Sphere() {}
Sphere::~Sphere() {}
std::shared_ptr<data::DataNode> Sphere::Serialize() const {
    auto tdb = base_type::Serialize();
    tdb->SetValue("Origin", m_origin_);
    tdb->SetValue("Radius", m_radius_);
    return tdb;
};
void Sphere::Deserialize(std::shared_ptr<const data::DataNode>const & cfg) {
    base_type::Deserialize(cfg);
    m_origin_ = db()->GetValue("Origin", m_origin_);
    m_radius_ = db()->GetValue("Radius", m_radius_);
}
}
}