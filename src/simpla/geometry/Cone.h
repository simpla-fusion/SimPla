//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONE_H
#define SIMPLA_CONE_H

#include <simpla/utilities/Constants.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "ParametricBody.h"
#include "ParametricSurface.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {

struct sfCone : public ShapeFunction {
    sfCone() = default;
    sfCone(sfCone const &) = default;
    ~sfCone() = default;
    explicit sfCone(Real semi_angle) { SetSemiAngle(semi_angle); }

    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;

    point_type Value(Real l, Real phi, Real theta) const override {
        Real r = l * std::sin(theta);
        return point_type{r * std::cos(phi), r * std::sin(phi), l * std::cos(theta)};
    }
    point_type Value(Real l, Real phi) const { return Value(l, phi, GetSemiAngle()); }
    point_type InvValue(Real x, Real y, Real z) const override { return point_type{x, y, z}; }

    point_type InvValue(point_type const &xyz) const override { return xyz; }
    Real Distance(point_type const &xyz) const override { return xyz[2]; }
    bool TestBoxGetIntersectionion(point_type const &x_min, point_type const &x_max) const override {
        return x_min[2] < 0 && x_max[2] > 0;
    }
    int LineGetIntersectionion(point_type const &p0, point_type const &p1, Real *u) const override { return 0; }

    void SetSemiAngle(Real theta) { m_parameter_range_[1][2] = theta; }
    Real GetSemiAngle() const { return m_parameter_range_[1][2]; }

   protected:
    Real m_parameter_range_[2][3] = {{0, 0, 0}, {1, TWOPI, PI / 4}};
    Real m_value_range_[2][3] = {{-1, -1, -1}, {1, 1, 1}};
};
struct ConicalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(ConicalSurface, ParametricSurface);

   protected:
    ConicalSurface();
    ConicalSurface(ConicalSurface const &other);
    ConicalSurface(Axis const &axis, Real semi_angle);

   public:
    ~ConicalSurface() override;

    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;

    point_type xyz(Real u, Real v) const override;
    point_type uvw(Real x, Real y, Real z) const override;

   private:
    sfCone m_shape_;
};

struct Cone : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Cone, ParametricBody)

   protected:
    Cone();
    explicit Cone(Axis const &axis, Real semi_angle);

   public:
    ~Cone() override;
    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;

    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

   private:
    sfCone m_shape_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
