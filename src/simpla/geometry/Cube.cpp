//
// Created by salmon on 17-5-31.
//
#include "Cube.h"
#include "GeoAlgorithm.h"
#include "Line.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Cube)

std::shared_ptr<data::DataNode> Cube::Serialize() const {
    auto cfg = base_type::Serialize();
    if (cfg != nullptr) { cfg->SetValue("Box", m_bound_box_); }
    return cfg;
};
void Cube::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);

    if (cfg != nullptr) {
        if (cfg->Get("Box") != nullptr) {
            m_bound_box_ = cfg->GetValue<box_type>("Box");
        } else {
            std::get<0>(m_bound_box_) = cfg->GetValue<nTuple<Real, 3>>("lo", std::get<0>(m_bound_box_));
            std::get<1>(m_bound_box_) = cfg->GetValue<nTuple<Real, 3>>("hi", std::get<1>(m_bound_box_));
        }
    }
}

std::shared_ptr<GeoObject> Cube::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    //    if (auto line = std::dynamic_pointer_cast<const Line>(g)) { res = PointsOnCurve::New(line); }
    return nullptr;
}
}  // namespace geometry {
}  // namespace simpla {