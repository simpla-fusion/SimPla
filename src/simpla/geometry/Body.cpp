//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "Surface.h"

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
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Curve> Body::Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Surface> Body::Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Body> Body::Intersection(std::shared_ptr<const Body> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool Body::TestIntersection(point_type const &p, Real tolerance) const { return TestPointInBox(p, GetBoundingBox()); };
bool Body::TestIntersection(box_type const &box, Real tolerance) const {
    return TestBoxOverlapped(box, GetBoundingBox());
};
box_type Body::GetBoundingBox() const {
    return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
};
std::shared_ptr<GeoObject> Body::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
        res = Intersection(curve, tolerance);
    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
        res = Intersection(surface, tolerance);
    } else if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
        res = Intersection(body, tolerance);
    } else {
        UNIMPLEMENTED;
    }
    return res;
};
bool Body::TestIntersection(box_type const &, Real tolerance) const { return false; };

// point_type Body::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
// point_type Body::xyz(point_type const &u) const { return xyz(u[0], u[1], u[2]); };
// point_type Body::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
// point_type Body::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };

// bool Body::TestIntersection(Real x, Real y, Real z, Real tolerance) const {
//    return TestPointInBox(point_type{x, y, z}, GetBoundingBox());
//};
// bool Body::TestIntersection(point_type const &xyz, Real tolerance) const {
//    return TestIntersection(xyz[0], xyz[1], xyz[2], tolerance);
//}

// bool Body::TestInsideUVW(Real u, Real v, Real w, Real tolerance) const {
//    return TestPointInBox(point_type{u, v, w}, GetParameterRange());
//}
// bool Body::TestInsideUVW(point_type const &uvw, Real tolerance) const {
//    return TestInsideUVW(uvw[0], uvw[1], uvw[2], tolerance);
//};

}  // namespace geometry
}  // namespace simpla