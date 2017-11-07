/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <vector>
#include "BoundedCurve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Polygon : public BoundedCurve2D {
    SP_GEO_OBJECT_HEAD(Polygon, BoundedCurve2D)
   public:
    Polygon(point_type const &p0, point_type const &p1);
   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
