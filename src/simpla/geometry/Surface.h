//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include "GeoObject.h"
#include "Axis.h"
namespace simpla {
namespace geometry {

/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 *
 */
struct Surface : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Surface, GeoObject);
    Surface() = default;
    Surface(Surface const &) = default;
    ~Surface() override = default;

    template <typename... Args>
    explicit Surface(Args &&... args) : m_axis_(std::forward<Args>(args)...) {}

    virtual std::tuple<bool, bool> IsClosed() const { return std::make_tuple(false, false); };
    virtual std::tuple<bool, bool> IsPeriodic() const { return std::make_tuple(false, false); };
    virtual nTuple<Real, 2> GetPeriod() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; };
    virtual nTuple<Real, 2> GetMinParameter() const { return nTuple<Real, 2>{-SP_INFINITY, -SP_INFINITY}; }
    virtual nTuple<Real, 2> GetMaxParameter() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }

    virtual point_type Value(Real u, Real v) const = 0;
    point_type Value(nTuple<Real, 2> const &u) const { return Value(u[0], u[1]); };

    void SetAxis(Axis const &a) { m_axis_ = a; }
    Axis const &GetAxis() const { return m_axis_; }

   protected:
    Axis m_axis_;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
