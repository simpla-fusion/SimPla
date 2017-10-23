//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include "Axis.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 *
 */
struct Surface : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Surface, GeoObject);
    Surface() = default;
    ~Surface() override = default;
    Surface(Surface const &other) : m_axis_(other.m_axis_) {}
    explicit Surface(std::shared_ptr<Axis> const &axis) : m_axis_(axis) {}

    virtual std::tuple<bool, bool> IsClosed() const { return std::make_tuple(false, false); };
    virtual std::tuple<bool, bool> IsPeriodic() const { return std::make_tuple(false, false); };
    virtual nTuple<Real, 2> GetPeriod() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; };
    virtual nTuple<Real, 2> GetMinParameter() const { return nTuple<Real, 2>{-SP_INFINITY, -SP_INFINITY}; }
    virtual nTuple<Real, 2> GetMaxParameter() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }

    void SetParameterRange(std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> const &r) {
        std::tie(m_uv_min_, m_uv_max_) = r;
    };
    void SetParameterRange(nTuple<Real, 2> const &min, nTuple<Real, 2> const &max) {
        m_uv_min_ = min;
        m_uv_max_ = max;
    };
    std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> GetParameterRange() const {
        return std::make_tuple(m_uv_min_, m_uv_max_);
    };

    virtual point_type Value(Real u, Real v) const = 0;

    point_type Value(nTuple<Real, 2> const &u) const { return Value(u[0], u[1]); };

    void SetAxis(std::shared_ptr<Axis> const &a) { m_axis_ = a; }
    std::shared_ptr<Axis> GetAxis() const { return m_axis_; }

    void Mirror(const point_type &p) override { m_axis_->Mirror(p); }
    void Mirror(const Axis &a1) override { m_axis_->Mirror(a1); }
    void Rotate(const Axis &a1, Real angle) override { m_axis_->Rotate(a1, angle); }
    void Scale(Real s, int dir) override { m_axis_->Scale(s); }
    void Translate(const vector_type &v) override { m_axis_->Translate(v); }

   protected:
    std::shared_ptr<Axis> m_axis_;
    nTuple<Real, 2> m_uv_min_{SP_SNaN, SP_SNaN};
    nTuple<Real, 2> m_uv_max_{SP_SNaN, SP_SNaN};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
