//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_SHAPE_H
#define SIMPLA_SHAPE_H

#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Body;
struct Shell;
struct Shape : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Shape, GeoObject)
    virtual std::shared_ptr<Body> AsBody() const = 0;
    virtual std::shared_ptr<Shell> AsShell() const = 0;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SHAPE_H
