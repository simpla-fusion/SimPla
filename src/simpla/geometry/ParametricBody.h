//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETRICBODY_H
#define SIMPLA_PARAMETRICBODY_H
#include "Body.h"
namespace simpla {
namespace geometry {
struct ParametricSurface;
struct ParametricBody : public Body {
    SP_GEO_ABS_OBJECT_HEAD(ParametricBody, Body);

   protected:
    ParametricBody();
    ParametricBody(ParametricBody const &other);
    explicit ParametricBody(Axis const &axis);

   public:
    ~ParametricBody() override;

    std::shared_ptr<Surface> GetBoundarySurface() const override;
    std::shared_ptr<Curve> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    std::shared_ptr<Surface> Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const override;
    std::shared_ptr<Body> Intersection(std::shared_ptr<const Body> const &g, Real tolerance) const override;
    bool TestIntersection(point_type const &, Real tolerance) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;
    box_type GetBoundingBox() const override;

    box_type const &GetParameterRange() const;
    box_type const &GetValueRange() const;

    virtual point_type xyz(Real u, Real v, Real w) const = 0;
    virtual point_type uvw(Real x, Real y, Real z) const = 0;
    point_type xyz(point_type const &u) const;
    point_type uvw(point_type const &x) const;

   protected:
    void SetParameterRange(box_type const &);
    void SetValueRange(box_type const &);

    box_type m_parameter_range_{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    box_type m_value_range_{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    std::shared_ptr<ParametricSurface> m_surface_ = nullptr;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETRICBODY_H
