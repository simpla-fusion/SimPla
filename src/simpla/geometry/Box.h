//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <simpla/SIMPLA_config.h>
#include "Body.h"
#include "ParametricBody.h"
namespace simpla {
namespace geometry {

struct Box : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Box, ParametricBody)

   protected:
    Box();
    Box(Box const &);
    Box(std::initializer_list<std::initializer_list<Real>> const &v);
    Box(point_type const &p0, point_type const &p1);
    explicit Box(box_type const &v);

   public:
    ~Box() override;
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }
    box_type GetParameterRange() const override;
    box_type GetValueRange() const override;
    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    bool CheckIntersection(box_type const &, Real tolerance) const override;
    //    std::shared_ptr<Point> GetIntersectionion(std::shared_ptr<const Point> const &g, Real tolerance) const override;
    //    std::shared_ptr<Curve> GetIntersectionion(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    //    std::shared_ptr<Surface> GetIntersectionion(std::shared_ptr<const Surface> const &g, Real tolerance) const override;
    //    std::shared_ptr<Body> GetIntersectionion(std::shared_ptr<const Body> const &g, Real tolerance) const override;

   protected:
    static constexpr Real m_parameter_range_[2][3] = {{0, 0, 0}, {1, 1, 1}};
    static constexpr Real m_value_range_[2][3] = {{0, 0, 0}, {1, 1, 1}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
