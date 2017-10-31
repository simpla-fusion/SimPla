//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"
#include "Curve.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Swept : public Body {
    SP_GEO_OBJECT_HEAD(Swept, Body);

   protected:
    Swept();
    Swept(Swept const &other);
    explicit Swept(Axis const &axis);

   public:
    ~Swept() override;

    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;
};

struct SweptSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(SweptSurface, Surface);

   protected:
    SweptSurface();
    SweptSurface(SweptSurface const &other);
    SweptSurface(Axis const &axis);

   public:
    ~SweptSurface() override;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTBODY_H
