//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Surface;
struct Curve;
struct Body : public GeoObject {
    SP_OBJECT_HEAD(Body, GeoObject)
    bool CheckInside(point_type const &x) const override;

    virtual std::shared_ptr<Surface> GetBoundary() const;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
