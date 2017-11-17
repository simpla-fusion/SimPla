//
// Created by salmon on 17-10-24.
//

#include "Torus.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Torus)
Torus::Torus() = default;
Torus::Torus(Torus const &) = default;
Torus::Torus(Axis const &axis) : PrimitiveShape(axis) {}
Torus::Torus(Axis const &axis, Real major_radius, Real minor_radius)
    : PrimitiveShape(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}
Torus::Torus(Real major_radius, Real minor_radius) : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}
Torus::~Torus() = default;

std::shared_ptr<simpla::data::DataEntry> Torus::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("MajorRadius", m_major_radius_);
    res->SetValue("MinorRadius", m_minor_radius_);
    return res;
};
void Torus::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue("MinorRadius", m_minor_radius_);
}

Real Torus::GetMajorRadius() const { return m_major_radius_; }
Real Torus::GetMinorRadius() const { return m_minor_radius_; }

}  // namespace geometry {
}  // namespace simpla {