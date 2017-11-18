//
// Created by salmon on 17-10-24.
//

#include "spTorus.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Torus)
spTorus::spTorus() = default;
spTorus::spTorus(spTorus const &) = default;
spTorus::spTorus(Real major_radius, Real minor_radius) : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}
spTorus::~spTorus() = default;

std::shared_ptr<simpla::data::DataEntry> spTorus::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("MajorRadius", m_major_radius_);
    res->SetValue("MinorRadius", m_minor_radius_);
    return res;
};
void spTorus::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue("MinorRadius", m_minor_radius_);
}

Real spTorus::GetMajorRadius() const { return m_major_radius_; }
Real spTorus::GetMinorRadius() const { return m_minor_radius_; }

}  // namespace geometry {
}  // namespace simpla {