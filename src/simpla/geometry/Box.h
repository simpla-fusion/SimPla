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
    SP_OBJECT_HEAD(Box, Body)

   protected:
    Box(std::initializer_list<std::initializer_list<Real>> const &v) : Body() {
        SetParameterRange(point_type(*v.begin()), point_type(*(v.begin() + 1)));
    }

   public:
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }

    point_type Value(Real u, Real v, Real w) const override { return m_axis_.Coordinates(u, v, w); };
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
