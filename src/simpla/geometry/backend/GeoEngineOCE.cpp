//
// Created by salmon on 17-11-1.
//

#include "GeoEngineOCE.h"
#include "../GeoObject.h"
namespace simpla {
namespace geometry {
REGISTER_CREATOR(GeoEngineOCE, oce);
GeoEngineOCE::GeoEngineOCE() = default;
GeoEngineOCE::~GeoEngineOCE() = default;
// std::shared_ptr<GeoObject> GeoEngineOCE::GetBoundaryInterface(std::shared_ptr<const GeoObject> const &) const {}
// bool GeoEngineOCE::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &, point_type const &x,
//                                              Real tolerance) const {}
// bool GeoEngineOCE::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &, box_type const &,
//                                              Real tolerance) const {}

std::shared_ptr<GeoObject> GeoEngineOCE::GetUnionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                           std::shared_ptr<const GeoObject> const &g1,
                                                           Real tolerance) const {
    DUMMY << "Union : " << g0->FancyTypeName() << " && " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetDifferenceInterface(std::shared_ptr<const GeoObject> const &g0,
                                                                std::shared_ptr<const GeoObject> const &g1,
                                                                Real tolerance) const {
    DUMMY << "Difference : " << g0->FancyTypeName() << " && " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetIntersectionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                                  std::shared_ptr<const GeoObject> const &g1,
                                                                  Real tolerance) const {
    DUMMY << "Intersection : " << g0->FancyTypeName() << " && " << g1->FancyTypeName();
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla