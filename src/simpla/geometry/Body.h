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
struct Point;
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
    int GetDimension() const override { return 3; }

    virtual std::shared_ptr<Surface> GetBoundarySurface() const;
    std::shared_ptr<GeoObject> GetBoundary() const final;
    virtual std::shared_ptr<Point> GetIntersection(std::shared_ptr<const Point> const &g, Real tolerance) const;
    virtual std::shared_ptr<Curve> GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const;
    virtual std::shared_ptr<Surface> GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const;
    virtual std::shared_ptr<Body> GetIntersection(std::shared_ptr<const Body> const &g, Real tolerance) const;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const final;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g) const;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
