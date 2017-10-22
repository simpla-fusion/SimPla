//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H
#include <memory>
#include "Face.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
class Curve;

/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 *
 */
struct Surface : public GeoObject {
    SP_GEO_OBJECT_HEAD(Surface, GeoObject)
    Surface() = default;
    ~Surface() = default;
    //    std::shared_ptr<GeoObject> GetBoundary() const override;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
