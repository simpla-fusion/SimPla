//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/utilities/Constants.h>
#include "Axis.h"
#include "GeoObject.h"
#include "Plane.h"

namespace simpla {
namespace geometry {
struct PolyPoints;
struct Curve : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Curve, GeoObject);

   public:
    Curve() = default;
    Curve(Curve const &other) = default;
    explicit Curve(Axis const &axis) : GeoObject(axis) {}

   public:
    ~Curve() override = default;

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
struct PointsOnCurve : public PolyPoints {
    SP_GEO_OBJECT_HEAD(PointsOnCurve, PolyPoints);

   protected:
    PointsOnCurve();
    explicit PointsOnCurve(std::shared_ptr<const Curve> const &);

   public:
    ~PointsOnCurve() override;
    Axis const &GetAxis() const override;
    std::shared_ptr<const Curve> GetBasisCurve() const;
    void SetBasisCurve(std::shared_ptr<const Curve> const &c);

    void PutU(Real);
    Real GetU(size_type i) const;
    point_type GetPoint(size_type i) const;
    std::vector<Real> const &data() const;
    std::vector<Real> &data();

    size_type size() const override;
    point_type Value(size_type i) const override;
    box_type GetBoundingBox() const override;

   private:
    std::shared_ptr<const Curve> m_curve_;
    std::vector<Real> m_data_;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
