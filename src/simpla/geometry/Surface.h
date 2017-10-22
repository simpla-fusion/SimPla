//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H
#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
class Curve;
struct Surface : public GeoObject {
    SP_OBJECT_HEAD(Surface, GeoObject)
    std::shared_ptr<GeoObject> GetBoundary() const override;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
