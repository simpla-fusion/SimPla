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
    box_type const &GetParameterRange() const override { return m_parameter_range_; };
    box_type const &GetValueRange() const override { return m_value_range_; }

    point_type InvValue(point_type const &xyz) const override { return xyz; }
    Real Distance(point_type const &xyz) const override { return xyz[2]; }
    bool TestBoxIntersection(point_type const &x_min, point_type const &x_max) const override {
        return x_min[2] < 0 && x_max[2] > 0;
    }
    int LineIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return 0; }

    const box_type m_parameter_range_{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
                                      {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    const box_type m_value_range_{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
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
    ShapeFunction const &shape() const override { return m_shape_; }
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
