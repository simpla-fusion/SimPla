//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_SHAPE_H
#define SIMPLA_SHAPE_H
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include <memory>
#include "GeoObject.h"
namespace simpla {
template <typename, int...>
struct nTuple;
namespace geometry {
struct Curve;
struct Surface;
struct Body : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Body, GeoObject);

   protected:
    Body();
    Body(Body const &other);
    explicit Body(Axis const &axis);

   public:
    ~Body() override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SHAPE_H
