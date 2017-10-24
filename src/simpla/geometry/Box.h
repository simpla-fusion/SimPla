//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include "Body.h"
#include "simpla/SIMPLA_config.h"
namespace simpla {
namespace geometry {

struct Box : public Body {
    SP_GEO_OBJECT_HEAD(Box, Body)

   protected:
    Box(std::initializer_list<std::initializer_list<Real>> const &v) : Body() {
        SetParameterRange(point_type(*v.begin()), point_type(*(v.begin() + 1)));
    }
    Box();
    Box(Box const &);

   public:
    ~Box();
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }

    point_type Value(Real u, Real v, Real w) const override { return m_axis_.Coordinates(u, v, w); };

    int CheckOverlap(box_type const &, Real tolerance) const override;
    int FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
