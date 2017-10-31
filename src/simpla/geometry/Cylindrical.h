//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "ParametricBody.h"
#include "ParametricSurface.h"
#include "ShapeFunction.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct sfCylindrical : public ShapeFunction {
    point_type Value(Real u, Real v, Real w) const override { return point_type{u, v, w}; }
    point_type InvValue(point_type const &xyz) const override { return xyz; }
    Real Distance(point_type const &xyz) const override { return xyz[2]; }
    bool TestBoxIntersection(point_type const &x_min, point_type const &x_max) const override {
        return x_min[2] < 0 && x_max[2] > 0;
    }
    int LineIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return 0; }

    const box_type m_parameter_range_{{0, -SP_INFINITY, 0}, {SP_INFINITY, SP_INFINITY, TWOPI}};
    const box_type m_value_range_{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
};

struct Cylindrical : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Cylindrical, ParametricBody)

   protected:
    Cylindrical() = default;
    template <typename... Args>
    explicit Cylindrical(Axis const &axis, Args &&... args) : ParametricBody(axis) {}

   public:
    ~Cylindrical() override = default;
    box_type const &GetParameterRange() const override { return m_shape_.m_parameter_range_; };
    box_type const &GetValueRange() const override { return m_shape_.m_value_range_; };
    point_type xyz(Real u, Real v, Real w) const override { return m_axis_.xyz(m_shape_.Value(u, v, w)); };
    point_type uvw(Real x, Real y, Real z) const override { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };

    std::shared_ptr<Body> Intersection(std::shared_ptr<const Body> const &, Real tolerance) const override;
    std::shared_ptr<Curve> Intersection(std::shared_ptr<const Curve> const &, Real tolerance) const override;
    std::shared_ptr<Surface> Intersection(std::shared_ptr<const Surface> const &, Real tolerance) const override;

   protected:
    sfCylindrical m_shape_;
};

struct CylindricalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(CylindricalSurface, ParametricSurface);

   protected:
    CylindricalSurface();
    CylindricalSurface(CylindricalSurface const &other);
    CylindricalSurface(Axis const &axis, Real radius);

   public:
    ~CylindricalSurface() override;

    Real GetRadius() const { return m_radius_; }

    point_type xyz(Real u, Real v) const override { return m_axis_.xyz(m_shape_.Value(u, v, m_radius_)); };
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    Real m_radius_ = 1.0;
    sfCylindrical m_shape_;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
