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
struct Curve;
struct Body : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Body, GeoObject);

   protected:
    Body() = default;
    Body(Body const &other) = default;
    explicit Body(Axis const &axis) : GeoObject(axis){};

   public:
    ~Body() override = default;
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
    bool CheckInside(point_type const &p, Real tolerance) const override {
        return CheckOverlap(std::make_tuple(point_type{p - tolerance}, point_type{p + tolerance}), tolerance) > 1;
    }

    /**
    * @return
    *  <= 0 no overlap
    *  == 1 partial overlap
    *  >  1 all inside
    */
    virtual int CheckOverlap(box_type const &, Real tolerance) const;
    /**
     *
     * @return <0 first point is outgoing
     *         >0 first point is incoming
     */
    virtual int FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const;

   protected:
    nTuple<Real, 3> m_uvw_min_{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY};
    nTuple<Real, 3> m_uvw_max_{SP_INFINITY, SP_INFINITY, SP_INFINITY};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
