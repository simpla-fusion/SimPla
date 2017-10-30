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
template <typename, int...>
struct nTuple;
namespace geometry {
struct Curve;
struct Surface;
struct Body : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Body, GeoObject);

   protected:
    Body();
    Body(Body const &other);
    explicit Body(Axis const &axis);

   public:
    ~Body() override;

    virtual std::tuple<bool, bool, bool> IsClosed() const;
    virtual std::tuple<bool, bool, bool> IsPeriodic() const;
    virtual nTuple<Real, 3> GetPeriod() const;
    virtual nTuple<Real, 3> GetMinParameter() const;
    virtual nTuple<Real, 3> GetMaxParameter() const;

    void SetParameterRange(std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> const &r);
    void SetParameterRange(nTuple<Real, 3> const &min, nTuple<Real, 3> const &max);
    std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> GetParameterRange() const;
    bool TestInsideUVW(point_type const &x) const override;
    box_type GetBoundingBox() const override;

    virtual point_type Value(Real u, Real v, Real w) const = 0;
    point_type Value(point_type const &uvw) const override { return Value(uvw[0], uvw[1], uvw[2]); }

   protected:
    nTuple<Real, 3> m_uvw_min_{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY};
    nTuple<Real, 3> m_uvw_max_{SP_INFINITY, SP_INFINITY, SP_INFINITY};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
