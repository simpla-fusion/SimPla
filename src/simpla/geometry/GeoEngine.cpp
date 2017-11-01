//
// Created by salmon on 17-11-1.
//
#include "GeoEngine.h"
#include "GeoAlgorithm.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
GeoEngine::GeoEngine() = default;
// GeoEngine::GeoEngine(GeoEngine const &) = default;
GeoEngine::~GeoEngine() = default;
GeoEngine &GeoEngine::global() { return SingletonHolder<GeoEngine>::instance(); }
std::shared_ptr<GeoObject> GeoEngine::GetBoundary(std::shared_ptr<const GeoObject> const &g) {
    return global().GetBoundaryInterface(g);
}
bool GeoEngine::CheckIntersection(std::shared_ptr<const GeoObject> const &g, point_type const &x, Real tolerance) {
    return global().CheckIntersectionInterface(g, x, tolerance);
}
bool GeoEngine::CheckIntersection(std::shared_ptr<const GeoObject> const &g, box_type const &b, Real tolerance) {
    return global().CheckIntersectionInterface(g, b, tolerance);
}

std::shared_ptr<GeoObject> GeoEngine::GetUnion(std::shared_ptr<const GeoObject> const &g0,
                                               std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return global().GetIntersectionInterface(g0, g1, tolerance);
}
std::shared_ptr<GeoObject> GeoEngine::GetDifference(std::shared_ptr<const GeoObject> const &g0,
                                                    std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return global().GetIntersectionInterface(g0, g1, tolerance);
}
std::shared_ptr<GeoObject> GeoEngine::GetIntersection(std::shared_ptr<const GeoObject> const &g0,
                                                      std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return global().GetIntersectionInterface(g0, g1, tolerance);
}

std::shared_ptr<GeoObject> GeoEngine::GetBoundaryInterface(std::shared_ptr<const GeoObject> const &) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool GeoEngine::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, point_type const &p,
                                           Real tolerance) const {
    return TestPointInBox(p, g->GetBoundingBox());
}
bool GeoEngine::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, box_type const &box,
                                           Real tolerance) const {
    return TestBoxOverlapped(box, g->GetBoundingBox());
}
std::shared_ptr<GeoObject> GeoEngine::GetUnionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                        std::shared_ptr<const GeoObject> const &g1,
                                                        Real tolerance) const {
    DUMMY << "Union :" << g0->FancyTypeName() << " || " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngine::GetDifferenceInterface(std::shared_ptr<const GeoObject> const &g0,
                                                             std::shared_ptr<const GeoObject> const &g1,
                                                             Real tolerance) const {
    DUMMY << "Difference : " << g0->FancyTypeName() << " - " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngine::GetIntersectionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                               std::shared_ptr<const GeoObject> const &g1,
                                                               Real tolerance) const {
    DUMMY << "Intersection : " << g0->FancyTypeName() << " && " << g1->FancyTypeName()  ;
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla