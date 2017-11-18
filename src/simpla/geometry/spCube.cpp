//
// Created by salmon on 17-11-18.
//

#include "spCube.h"

namespace simpla {
namespace geometry {
spCube::spCube() = default;
spCube::spCube(spCube const &) = default;
spCube::~spCube() = default;

void spCube::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataEntry> spCube::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}

// std::shared_ptr<GeoObject> spCube::GetIntersection(std::shared_ptr<const GeoObject> const &g,
//                                                          Real tolerance) const {
//    std::shared_ptr<GeoObject> res = nullptr;
//    if (g == nullptr) {
//    } else if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
//        //        auto p_on_curve = PointsOnCurve::New(curve);
//        //        if (auto line = std::dynamic_pointer_cast<const spLine>(curve)) {
//        //            GetIntersectionLineCube(line->GetAxis().o, line->GetAxis().o + line->GetAxis().x, GetAxis().o,
//        //            m_radius_,
//        //                                tolerance, p_on_curve->data());
//        //        } else if (auto circle = std::dynamic_pointer_cast<const spCircle>(g)) {
//        //            UNIMPLEMENTED;
//        //        }
//        //        res = p_on_curve;
//    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
//        UNIMPLEMENTED;
//    } else if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
//        res = body->GetIntersection(std::dynamic_pointer_cast<const GeoObject>(shared_from_this()), tolerance);
//    }
//    return res;
//}
}  // namespace geometry {
}  // namespace simpla {
