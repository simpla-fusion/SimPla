//
// Created by salmon on 17-10-18.
//

#include "IntersectionCurveSurface.h"
#include <simpla/utilities/SPDefines.h>
#include "Body.h"
namespace simpla {
namespace geometry {

Intersector::Intersector(){};
Intersector::~Intersector(){};

std::shared_ptr<Intersector> Intersector::New(std::shared_ptr<const GeoObject> const& geo, Real tolerance) {
    if(auto g=std::dynamic_pointer_cast<Body>(geo))
    {

    }

}

void IntersectionCurveSurface::Load(std::shared_ptr<const GeoObject> const& body, Real tolerance) {
    m_pimpl_->m_geo_body_ = body;
    m_pimpl_->m_tolerance_ = tolerance;
}
template <typename TCurve>
IntersectionCurveSurface0() void IntersectionCurveSurface::Intersect(std::shared_ptr<const Curve> const& curve,
                                                                     std::vector<Real>& intersection_point) {}
}  // namespace geometry {
}  // namespace simpla