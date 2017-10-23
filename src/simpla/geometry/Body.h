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
namespace geometry {

struct Body : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Body, GeoObject);
    Body() : m_axis_(Axis::New()){};
    ~Body() override = default;

    Body(Body const &other) : m_axis_(other.m_axis_){};
    explicit Body(std::shared_ptr<Axis> const &axis) : m_axis_(axis){};

    virtual std::tuple<bool, bool, bool> IsClosed() const { return std::make_tuple(false, false, false); };
    virtual std::tuple<bool, bool, bool> IsPeriodic() const { return std::make_tuple(false, false, false); };
    virtual nTuple<Real, 3> GetPeriod() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, SP_INFINITY}; };
    virtual nTuple<Real, 3> GetMinParameter() const {
        return nTuple<Real, 3>{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY};
    }
    virtual nTuple<Real, 3> GetMaxParameter() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, SP_INFINITY}; }

    void SetParameterRange(std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> const &r) {
        std::tie(m_uvw_min_, m_uvw_max_) = r;
    };
    void SetParameterRange(nTuple<Real, 3> const &min, nTuple<Real, 3> const &max) {
        m_uvw_min_ = min;
        m_uvw_max_ = max;
    };
    std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> GetParameterRange() const {
        return std::make_tuple(m_uvw_min_, m_uvw_max_);
    };
    virtual point_type Value(Real u, Real v, Real w) const = 0;

    point_type Value(nTuple<Real, 3> const &u) const { return Value(u[0], u[1], u[2]); };

    box_type GetBoundingBox() const override {
        auto r = GetParameterRange();
        return std::make_tuple(Value(std::get<0>(r)), Value(std::get<1>(r)));
    };

    bool CheckInside(point_type const &x, Real tolerance) const override {
        auto uvw = m_axis_->uvw(x);
        return m_uvw_min_[0] <= x[0] && x[0] < m_uvw_max_[0] &&  //
               m_uvw_min_[1] <= x[1] && x[1] < m_uvw_max_[1] &&  //
               m_uvw_min_[2] <= x[2] && x[2] < m_uvw_max_[2];
    }
    void SetAxis(std::shared_ptr<Axis> const &a) { m_axis_ = a; }
    std::shared_ptr<Axis> GetAxis() const { return m_axis_; }

    void Mirror(const point_type &p) override { m_axis_->Mirror(p); }
    void Mirror(const Axis &a1) override { m_axis_->Mirror(a1); }
    void Rotate(const Axis &a1, Real angle) override { m_axis_->Rotate(a1, angle); }
    void Scale(Real s, int dir) override { m_axis_->Scale(s); }
    void Translate(const vector_type &v) override { m_axis_->Translate(v); }

   protected:
    std::shared_ptr<Axis> m_axis_ = nullptr;
    nTuple<Real, 3> m_uvw_min_{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY};
    nTuple<Real, 3> m_uvw_max_{SP_INFINITY, SP_INFINITY, SP_INFINITY};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
