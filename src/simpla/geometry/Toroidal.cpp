//
// Created by salmon on 17-10-24.
//

#include "Toroidal.h"
namespace simpla {
namespace geometry {
Toroidal::Toroidal() = default;
Toroidal::Toroidal(Toroidal const &) = default;
Toroidal::Toroidal(Axis const &axis) : ParametricBody(axis) {}

Toroidal::~Toroidal() = default;

std::shared_ptr<simpla::data::DataNode> Toroidal::Serialize() const {
    auto res = base_type::Serialize();
    //    res->SetValue("MajorRadius", m_major_radius_);
    return res;
};
void Toroidal::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    //    m_major_radius_ = cfg->GetValue("MajorRadius", m_major_radius_);
}

bool Toroidal::TestIntersection(box_type const &, Real tolerance) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> Toroidal::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}

/********************************************************************************************************************/

SP_OBJECT_REGISTER(ToroidalSurface)

void ToroidalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    //    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    //    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataNode> ToroidalSurface::Serialize() const {
    auto res = base_type::Serialize();
    //    res->SetValue<Real>("MajorRadius", m_major_radius_);
    //    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}
bool ToroidalSurface::TestIntersection(box_type const &, Real tolerance) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> ToroidalSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                         Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry {
}  // namespace simpla {