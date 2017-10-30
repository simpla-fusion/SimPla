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
struct Curve;
/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 *
 */
struct Surface : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Surface, GeoObject);
    Surface();
    ~Surface() override;
    Surface(Surface const &other);
    explicit Surface(Axis const &axis);

    virtual std::tuple<bool, bool> IsClosed() const;
    virtual std::tuple<bool, bool> IsPeriodic() const;
    virtual nTuple<Real, 2> GetPeriod() const;
    virtual nTuple<Real, 2> GetMinParameter() const;
    virtual nTuple<Real, 2> GetMaxParameter() const;

    void SetParameterRange(std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> const &r);
    void SetParameterRange(nTuple<Real, 2> const &min, nTuple<Real, 2> const &max);
    std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> GetParameterRange() const;

    virtual point_type Value(Real u, Real v) const = 0;
    point_type Value(nTuple<Real, 2> const &u) const;

    std::shared_ptr<GeoObject> GetBoundary() const override;
    box_type GetBoundingBox() const override;

    virtual bool TestInsideUV(Real u, Real v, Real tolerance) const;
    bool TestInsideUVW(point_type const &uvw, Real tolerance) const override;

    virtual bool TestInside(Real x, Real y, Real z, Real tolerance) const;
    bool TestInside(point_type const &x, Real tolerance) const override;

    bool TestIntersection(box_type const &) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;

    point_type Value(point_type const &uvw) const override;

   protected:
    nTuple<Real, 2> m_uv_min_{SP_SNaN, SP_SNaN};
    nTuple<Real, 2> m_uv_max_{SP_SNaN, SP_SNaN};
};
struct PointsOnSurface : public PolyPoints {
    SP_GEO_OBJECT_HEAD(PointsOnSurface, PolyPoints);

   protected:
    PointsOnSurface();
    explicit PointsOnSurface(std::shared_ptr<const Surface> const &);

   public:
    ~PointsOnSurface() override;
    std::shared_ptr<const Surface> GetBasisSurface() const;
    void SetBasisSurface(std::shared_ptr<const Surface> const &c);

    void PutUV(nTuple<Real, 2> uv);
    nTuple<Real, 2> GetUV(size_type i) const;
    point_type GetPoint(size_type i) const;
    std::vector<nTuple<Real, 2>> const &data() const;
    std::vector<nTuple<Real, 2>> &data();

    size_type size() const override;
    point_type Value(size_type i) const override;
    box_type GetBoundingBox() const override;

   private:
    std::shared_ptr<const Surface> m_surface_;
    std::vector<nTuple<Real, 2>> m_data_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
