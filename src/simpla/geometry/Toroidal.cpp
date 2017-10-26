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

int Toroidal::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Toroidal::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return 0;
}
}  // namespace geometry {
}  // namespace simpla {