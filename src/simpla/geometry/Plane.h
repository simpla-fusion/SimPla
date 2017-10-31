//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "ParametricSurface.h"
#include "ShapeFunction.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct sfPlane : public ShapeFunction {
    int GetDimension() const override { return 2; }
    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;

    point_type Value(Real u, Real v, Real w) const override { return point_type{u, v, w}; }
    point_type InvValue(Real x, Real y, Real z) const override { return point_type{x, y, z}; }

    Real Distance(point_type const &xyz) const override { return xyz[2]; }
    bool TestBoxIntersection(point_type const &x_min, point_type const &x_max) const override {
        return x_min[2] < 0 && x_max[2] > 0;
    }
    int LineIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return 0; }

    static constexpr Real m_parameter_range_[2][3] = {{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
                                                      {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    static constexpr Real m_value_range_[2][3] = {{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
                                                  {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
};

struct Plane : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(Plane, ParametricSurface);

   protected:
    Plane();
    Plane(Plane const &);
    explicit Plane(Axis const &axis);
    Plane(point_type const &o, vector_type const &x, vector_type const &y);

   public:
    ~Plane() override;

    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;
    point_type xyz(Real u, Real v) const override { return m_axis_.xyz(m_shape_.Value(u, v, 0)); }
    point_type uvw(Real x, Real y, Real z) const override { return m_axis_.uvw(m_shape_.InvValue(x, y, z)); }
    std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    std::shared_ptr<Curve> Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const override;
    bool TestIntersection(point_type const &, Real tolerance) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;

   protected:
    const sfPlane m_shape_;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
