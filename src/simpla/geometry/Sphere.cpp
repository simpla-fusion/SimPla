//
// Created by salmon on 17-10-23.
//

#include "Sphere.h"

#include <simpla/utilities/SPDefines.h>
#include "Box.h"
#include "Circle.h"
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "Line.h"

namespace simpla {
namespace geometry {
std::shared_ptr<GeoObject> Sphere::GetBoundary() const {
    return nullptr;  // std::dynamic_pointer_cast<GeoObject>(SphericalSurface::New(m_axis_, 1));
}
std::shared_ptr<simpla::data::DataNode> Sphere::Serialize() const { return base_type::Serialize(); };
void Sphere::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

bool Sphere::TestIntersection(box_type const &b, Real tolerance) const {
    //    return TestIntersectionCubeSphere(std::get<0>(b), std::get<1>(b), m_axis_.o, m_uvw_max_[0]);
    return false;
}

std::shared_ptr<GeoObject> Sphere::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g == nullptr || !TestIntersection(g->GetBoundingBox(), tolerance)) {
    } else if (auto line = std::dynamic_pointer_cast<const Line>(g)) {
        UNIMPLEMENTED;
    } else if (auto circle = std::dynamic_pointer_cast<const Circle>(g)) {
        UNIMPLEMENTED;
    } else if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
        UNIMPLEMENTED;
    } else if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
    }
    return res;
}
/**********************************************************************************************************************/

SP_OBJECT_REGISTER(SphericalSurface)
SphericalSurface::SphericalSurface(Axis const &axis, Real radius, Real phi0, Real phi1, Real theta0, Real theta1)
    : ParametricSurface(axis), m_radius_(radius) {}

void SphericalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> SphericalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
bool SphericalSurface::TestIntersection(box_type const &b, Real tolerance) const {
    return TestIntersectionCubeSphere(std::get<0>(b), std::get<1>(b), m_axis_.o, m_radius_);
}
bool SphericalSurface::TestIntersection(point_type const &x, Real tolerance) const {
    return dot(x - m_axis_.o, x - m_axis_.o) < m_radius_ * m_radius_;
}
std::shared_ptr<GeoObject> SphericalSurface::Intersection(std::shared_ptr<const GeoObject> const &g,
                                                          Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g == nullptr) {
    } else if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
        //        auto p_on_curve = PointsOnCurve::New(curve);
        //        if (auto line = std::dynamic_pointer_cast<const Line>(curve)) {
        //            IntersectLineSphere(line->GetAxis().o, line->GetAxis().o + line->GetAxis().x, GetAxis().o,
        //            m_radius_,
        //                                tolerance, p_on_curve->data());
        //        } else if (auto circle = std::dynamic_pointer_cast<const Circle>(g)) {
        //            UNIMPLEMENTED;
        //        }
        //        res = p_on_curve;
    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
        UNIMPLEMENTED;
    } else if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
        res = body->Intersection(std::dynamic_pointer_cast<const GeoObject>(shared_from_this()), tolerance);
    }
    return res;
}
}  // namespace geometry {
}  // namespace simpla {
