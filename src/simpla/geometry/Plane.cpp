//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include "GeoAlgorithm.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Plane::Serialize() const {
    auto cfg = base_type::Serialize();
    cfg->SetValue("Vertices", m_p);
    return cfg;
};
void Plane::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_p = cfg->GetValue("Vertices", m_p);
}
box_type Plane::GetBoundingBox() const {
    UNIMPLEMENTED;
    box_type res{m_p[0], m_p[1]};
    //    extent_box(&std::get<0>(res), &std::get<1>(res), m_p[2]);
    return res;
};
// bool Plane::CheckInside(point_type const& x, Real tolerance) const { return false; };
}  // namespace geometry
}  // namespace simpla