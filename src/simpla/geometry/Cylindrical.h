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

/**
 *  R phi Z
 */
struct sfCylindrical : public ShapeFunction {
    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;

    point_type Value(Real u, Real v, Real w) const override { return point_type{u, v, w}; }
    point_type InvValue(point_type const &xyz) const override { return xyz; }
    Real Distance(point_type const &xyz) const override { return xyz[2]; }
    bool TestBoxIntersection(point_type const &x_min, point_type const &x_max) const override {
        return x_min[2] < 0 && x_max[2] > 0;
    }
    int LineIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return 0; }

   protected:
    static constexpr Real m_parameter_range_[2][3] = {{0, -SP_INFINITY, 0}, {1, TWOPI, 1}};
    static constexpr Real m_value_range_[2][3] = {{-1, -1, -1}, {1, 1, 1}};
};

struct Cylindrical : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Cylindrical, ParametricBody)

   protected:
    Cylindrical() = default;
    template <typename... Args>
    explicit Cylindrical(Axis const &axis, Args &&... args) : ParametricBody(axis) {}

   public:
    ~Cylindrical() override = default;
    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;
    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    bool TestIntersection(point_type const &p, Real tolerance) const override;
    bool TestIntersection(box_type const &b, Real tolerance) const override;

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
    explicit CylindricalSurface(Axis const &axis, Real radius);

   public:
    ~CylindricalSurface() override;

    Real GetRadius() const { return m_radius_; }

    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;
    point_type xyz(Real phi, Real Z) const override { return m_axis_.xyz(m_shape_.Value(m_radius_, phi, Z)); };
    bool TestIntersection(box_type const &, Real tolerance) const override;

    std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    std::shared_ptr<Curve> Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const override;

   private:
    Real m_radius_ = 1.0;
    sfCylindrical m_shape_;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
