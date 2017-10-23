//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_LINEAREXTRUSIONSURFACE_H
#define SIMPLA_LINEAREXTRUSIONSURFACE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"
#include "SweptSurface.h"
namespace simpla {
namespace geometry {
struct LinearExtrusionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(LinearExtrusionSurface, SweptSurface);

   protected:
    LinearExtrusionSurface() = default;
    LinearExtrusionSurface(LinearExtrusionSurface const &other) = default;  // : SweptSurface(other) {}
    LinearExtrusionSurface(std::shared_ptr<Curve> const &base, vector_type const &shift) : SweptSurface(base) {
        SetParameterRange(GetMinParameter(), GetMaxParameter());
    }

   public:
    ~LinearExtrusionSurface() override = default;

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

   protected:
    vector_type m_shift_;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_LINEAREXTRUSIONSURFACE_H
