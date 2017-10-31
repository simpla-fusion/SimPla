//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETRICSHAPE_H
#define SIMPLA_PARAMETRICSHAPE_H

#include "Body.h"
#include "Curve.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct ParametricCurve : public Curve {
    SP_GEO_ABS_OBJECT_HEAD(ParametricCurve, Curve);

   public:
    ParametricCurve() = default;
    ParametricCurve(Curve const &other) = default;
    explicit ParametricCurve(Axis const &axis) : GeoObject(axis) {}

   public:
    ~ParametricCurve() override = default;

    virtual bool IsClosed() const { return false; };
    virtual bool IsPeriodic() const { return false; };
    virtual Real GetPeriod() const { return SP_INFINITY; };
    virtual Real GetMinParameter() const { return -SP_INFINITY; }
    virtual Real GetMaxParameter() const { return SP_INFINITY; }
    void SetParameterRange(std::tuple<Real, Real> const &r) { std::tie(m_u_min_, m_u_max_) = r; };
    void SetParameterRange(Real min, Real max) {
        m_u_min_ = min;
        m_u_max_ = max;
    };
    std::tuple<Real, Real> GetParameterRange() const {
        return std::make_tuple(std::isnan(m_u_min_) ? GetMinParameter() : m_u_min_,
                               std::isnan(m_u_max_) ? GetMaxParameter() : m_u_max_);
    };

    virtual bool IsOnPlane() const { return false; };

    point_type StartPoint() const { return Value(m_u_min_); }
    point_type EndPoint() const { return Value(m_u_max_); }

    virtual point_type Value(Real u) const = 0;
    bool TestInsideU(Real u) const;
    std::shared_ptr<GeoObject> GetBoundary() const override;
    box_type GetBoundingBox() const override;
    bool TestIntersection(box_type const &) const override;
    bool TestInside(point_type const &x, Real tolerance) const override;
    bool TestInsideUVW(point_type const &x, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    point_type Value(point_type const &uvw) const override;

   protected:
    Real m_u_min_ = SP_SNaN, m_u_max_ = SP_SNaN;
};

struct ParametricSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(ParametricSurface, GeoObject);
    ParametricSurface();
    ~ParametricSurface() override;
    ParametricSurface(Surface const &other);
    explicit ParametricSurface(Axis const &axis);

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
struct ParametricSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(ParametricSurface, Surface)
   protected:
    ParametricSurface();
    ParametricSurface(ParametricSurface const &other);
    explicit ParametricSurface(Axis const &axis);

   public:
    ~ParametricSurface() override = default;
    virtual point_type xyz(Real u, Real v, Real w) const = 0;
    virtual point_type uvw(Real x, Real y, Real z) const = 0;

    virtual point_type xyz(point_type const &uvw) const = 0;
    virtual point_type uvw(point_type const &xyz) const { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; };
};
 struct ParametricBody : public Body {
    virtual bool TestInsideUVW(Real u, Real v, Real w, Real tolerance) const;
    bool TestInsideUVW(point_type const &x, Real tolerance) const override;

    virtual box_type GetParameterRange() const;
    virtual box_type GetValueRange() const;

    virtual point_type xyz(Real u, Real v, Real w) const;
    virtual point_type uvw(Real x, Real y, Real z) const;

    point_type xyz(point_type const &u) const override;
    point_type uvw(point_type const &x) const override;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETRICSHAPE_H
