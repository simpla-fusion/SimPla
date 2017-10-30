//
// Created by salmon on 17-10-24.
//

#include "Toroidal.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Toroidal::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("MajorRadius", m_major_radius_);
    return res;
};
void Toroidal::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue("MajorRadius", m_major_radius_);
}

bool Toroidal::TestIntersection(box_type const &) const { return false; }
std::shared_ptr<GeoObject> Toroidal::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}

/********************************************************************************************************************/

SP_OBJECT_REGISTER(ToroidalSurface)

void ToroidalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataNode> ToroidalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MajorRadius", m_major_radius_);
    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}
bool ToroidalSurface::TestIntersection(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> ToroidalSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                         Real tolerance) const {
    return nullptr;
}
}  // namespace geometry {
}  // namespace simpla {