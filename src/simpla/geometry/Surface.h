//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include <utility>
#include "Axis.h"
#include "GeoObject.h"
#include "PolyPoints.h"

namespace simpla {
template <typename, int...>
struct nTuple;
namespace geometry {
struct PointsOnCurve;
struct Curve;
struct Body;
/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 */

struct Surface : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Surface, GeoObject);
   public:
    virtual std::shared_ptr<PointsOnCurve> GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const;
    virtual std::shared_ptr<Curve> GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const final;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g) const;
};
//
// struct PointsOnSurface : public PolyPoints {
//    SP_GEO_OBJECT_HEAD(PointsOnSurface, PolyPoints);
//
//   protected:
//    PointsOnSurface();
//    explicit PointsOnSurface(std::shared_ptr<const Surface> const &);
//
//   public:
//    ~PointsOnSurface() override;
//    std::shared_ptr<const Surface> GetBasisSurface() const;
//    void SetBasisSurface(std::shared_ptr<const Surface> const &c);
//
//    void PutUV(nTuple<Real, 2> uv);
//    nTuple<Real, 2> GetUV(size_type i) const;
//    point_type GetPoint(size_type i) const;
//    std::vector<nTuple<Real, 2>> const &data() const;
//    std::vector<nTuple<Real, 2>> &data();
//
//    size_type size() const override;
//    point_type Value(size_type i) const override;
//    box_type GetBoundingBox() const override;
//
//   private:
//    std::shared_ptr<const Surface> m_curve_;
//    std::vector<nTuple<Real, 2>> m_data_;
//};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
