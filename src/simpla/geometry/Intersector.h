//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_INTERSECTIONCURVESURFACE_H
#define SIMPLA_INTERSECTIONCURVESURFACE_H

#include <simpla/data/SPObject.h>
#include <simpla/utilities/Log.h>
#include <memory>
#include <vector>
#include "simpla/SIMPLA_config.h"

namespace simpla {
namespace geometry {
class GeoObject;
class Curve;
class Surface;
class Body;
struct Intersector : public SPObject {
    SP_OBJECT_HEAD(Intersector, SPObject)
   protected:
    Intersector(std::shared_ptr<const GeoObject> const& geo, Real tolerance);

   public:
    static std::shared_ptr<Intersector> New(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);
    void SetGeoObject(std::shared_ptr<const GeoObject> const& geo);
    std::shared_ptr<const GeoObject> SetGeoObject() const;
    void SetTolerance(Real tolerance);
    Real GetTolerance() const;

    virtual size_type GetIntersectionPoints(std::shared_ptr<const GeoObject> const& line,
                                            std::vector<Real>& intersection_point) const;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_INTERSECTIONCURVESURFACE_H
