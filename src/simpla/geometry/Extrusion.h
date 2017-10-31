//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_LINEAREXTRUSIONBODY_H
#define SIMPLA_LINEAREXTRUSIONBODY_H

#include <simpla/utilities/Constants.h>
#include "Swept.h"
namespace simpla {
namespace geometry {
struct Curve;
struct Extrusion : public Swept {
    SP_GEO_OBJECT_HEAD(Extrusion, Swept);

   protected:
    Extrusion();
    Extrusion(Extrusion const &other);
    Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &c);
    bool TestIntersection(box_type const &) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   public:
    ~Extrusion() override;
};
struct ExtrusionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(ExtrusionSurface, SweptSurface);

   protected:
    ExtrusionSurface() = default;
    ExtrusionSurface(ExtrusionSurface const &other) = default;  // : SweptSurface(other) {}
    ExtrusionSurface(std::shared_ptr<Curve> const &base, vector_type const &shift) : SweptSurface(base) {
        SetParameterRange(GetMinParameter(), GetMaxParameter());
    }

   public:
    ~ExtrusionSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(GetBasisCurve()->IsClosed(), false); };
    std::tuple<bool, bool> IsPeriodic() const override {
        return std::make_tuple(GetBasisCurve()->IsPeriodic(), false);
    };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{GetBasisCurve()->GetPeriod(), SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override {
        return nTuple<Real, 2>{GetBasisCurve()->GetMinParameter(), -SP_INFINITY};
    }
    nTuple<Real, 2> GetMaxParameter() const override {
        return nTuple<Real, 2>{GetBasisCurve()->GetMaxParameter(), SP_INFINITY};
    }

    point_type Value(Real u, Real v) const override { return GetBasisCurve()->Value(u) + v * m_shift_; };
    bool TestIntersection(box_type const &) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   protected:
    vector_type m_shift_;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_LINEAREXTRUSIONBODY_H
