//
// Created by salmon on 17-7-22.
//

#include "Cylindrical.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Cylindrical::Serialize() const { return base_type::Serialize(); };
void Cylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

int Cylindrical::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Cylindrical::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return 0;
}

/**********************************************************************************************************************/

SP_OBJECT_REGISTER(CylindricalSurface)

void CylindricalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> CylindricalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
int CylindricalSurface::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> CylindricalSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                            Real tolerance) const {
    return nullptr;
}
}  // namespace geometry {
}  // namespace simpla {
