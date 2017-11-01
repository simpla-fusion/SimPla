//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "Surface.h"
#include "Point.h"

namespace simpla {
namespace geometry {
Body::Body() = default;
Body::Body(Body const &other) = default;
Body::Body(Axis const &axis) : GeoObject(axis) {}
Body::~Body() = default;

std::shared_ptr<data::DataNode> Body::Serialize() const { return base_type::Serialize(); };
void Body::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

std::shared_ptr<GeoObject> Body::GetBoundary() const { return GetBoundarySurface(); };
std::shared_ptr<Surface> Body::GetBoundarySurface() const {
    return std::dynamic_pointer_cast<Surface>(base_type::GetBoundary());
}
std::shared_ptr<Point> Body::GetIntersection(std::shared_ptr<const Point> const &g, Real tolerance) const {
    return std::dynamic_pointer_cast<Point>(base_type::GetIntersection((g), tolerance));
};
std::shared_ptr<Curve> Body::GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
    return std::dynamic_pointer_cast<Curve>(base_type::GetIntersection((g), tolerance));
}
std::shared_ptr<Surface> Body::GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
    return std::dynamic_pointer_cast<Surface>(base_type::GetIntersection(g, tolerance));
}
std::shared_ptr<Body> Body::GetIntersection(std::shared_ptr<const Body> const &g, Real tolerance) const {
    return std::dynamic_pointer_cast<Body>(base_type::GetIntersection(g, tolerance));
}

std::shared_ptr<GeoObject> Body::GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
        res = GetIntersection(curve, tolerance);
    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
        res = GetIntersection(surface, tolerance);
    } else if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
        res = GetIntersection(body, tolerance);
    }
    if (res == nullptr) { res = base_type::GetIntersection(g, tolerance); }
    return res;
};
std::shared_ptr<GeoObject> Body::GetIntersection(std::shared_ptr<const GeoObject> const &g) const {
    return GetIntersection(g, SP_GEO_DEFAULT_TOLERANCE);
}

// point_type Body::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
// point_type Body::xyz(point_type const &u) const { return xyz(u[0], u[1], u[2]); };
// point_type Body::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
// point_type Body::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };

// bool Body::CheckIntersection(Real x, Real y, Real z, Real tolerance) const {
//    return TestPointInBox(point_type{x, y, z}, GetBoundingBox());
//};
// bool Body::CheckIntersection(point_type const &xyz, Real tolerance) const {
//    return CheckIntersection(xyz[0], xyz[1], xyz[2], tolerance);
//}

// bool Body::TestInsideUVW(Real u, Real v, Real w, Real tolerance) const {
//    return TestPointInBox(point_type{u, v, w}, GetParameterRange());
//}
// bool Body::TestInsideUVW(point_type const &uvw, Real tolerance) const {
//    return TestInsideUVW(uvw[0], uvw[1], uvw[2], tolerance);
//};

}  // namespace geometry
}  // namespace simpla