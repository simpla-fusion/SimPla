//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

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

    virtual std::shared_ptr<Surface> GetBoundarySurface() const;
    virtual std::shared_ptr<Curve> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const;
    virtual std::shared_ptr<Surface> Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const;
    virtual std::shared_ptr<Body> Intersection(std::shared_ptr<const Body> const &g, Real tolerance) const;
    bool TestIntersection(point_type const &, Real tolerance) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;
    box_type GetBoundingBox() const override;
    std::shared_ptr<GeoObject> GetBoundary() const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
