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
    explicit Curve(Axis const &axis) : m_axis_(axis) {}

   public:
    ~Curve() override = default;

    virtual bool IsClosed() const { return false; };
    virtual bool IsPeriodic() const { return false; };
    virtual Real GetPeriod() const { return SP_INFINITY; };
    virtual Real GetMinParameter() const { return -SP_INFINITY; }
    virtual Real GetMaxParameter() const { return SP_INFINITY; }

    virtual point_type Value(Real u) const = 0;

    void SetAxis(Axis const &a) { m_axis_ = a; }
    Axis const &GetAxis() const { return m_axis_; }

   protected:
    Axis m_axis_;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
