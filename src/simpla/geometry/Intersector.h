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
   protected:
    Intersector();

   public:
    ~Intersector();

    static std::shared_ptr<Intersector> New(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);

    virtual size_type GetIntersectionPoints(std::shared_ptr<const Curve> const& curve,
                                            std::vector<Real>& intersection_point) const;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_INTERSECTIONCURVESURFACE_H
