//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/utilities/Constants.h>
#include "Axis.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Curve : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Curve, GeoObject);

   public:
    Curve() = default;
    Curve(Curve const &other) : m_axis_(other.m_axis_){};
    explicit Curve( Axis  const &axis) : m_axis_(axis) {}

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

    virtual point_type Value(Real u) const = 0;

    void SetAxis( Axis  const &a) { m_axis_ = a; }
     Axis  GetAxis() const { return m_axis_; }

   protected:
     Axis  m_axis_;
    Real m_u_min_ = SP_SNaN, m_u_max_ = SP_SNaN;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
