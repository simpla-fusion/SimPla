//
// Created by salmon on 17-10-23.
//

#include "spSphere.h"

#include <simpla/utilities/SPDefines.h>
#include "Box.h"
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "spLine.h"
#include "spCircle.h"

namespace simpla {
namespace geometry {
SP_SHAPE_REGISTER(spSphere)
spSphere::spSphere() = default;
spSphere::spSphere(spSphere const &) = default;
spSphere::~spSphere() = default;
spSphere::spSphere(Real radius) : m_radius_(radius) {}

void spSphere::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataEntry> spSphere::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}

// bool spSphere::CheckIntersection(box_type const &b, Real tolerance) const {
//    return CheckIntersectionCubeSphere(std::get<0>(b), std::get<1>(b), m_axis_.o, m_radius_);
//}
// bool spSphere::CheckIntersection(point_type const &x, Real tolerance) const {
//    return dot(x - m_axis_.o, x - m_axis_.o) < m_radius_ * m_radius_;
//}
// std::shared_ptr<GeoObject> spSphere::GetIntersection(std::shared_ptr<const GeoObject> const &g,
//                                                          Real tolerance) const {
//    std::shared_ptr<GeoObject> res = nullptr;
//    if (g == nullptr) {
//    } else if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
//        //        auto p_on_curve = PointsOnCurve::New(curve);
//        //        if (auto line = std::dynamic_pointer_cast<const spLine>(curve)) {
//        //            GetIntersectionLineSphere(line->GetAxis().o, line->GetAxis().o + line->GetAxis().x, GetAxis().o,
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
