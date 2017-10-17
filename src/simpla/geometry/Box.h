//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include "simpla/SIMPLA_config.h"
#include "GeoObject.h"

namespace simpla {
namespace geometry {

struct Box : public GeoObject {
    SP_OBJECT_HEAD(Box, GeoObject)

    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};

   protected:
    Box(std::initializer_list<std::initializer_list<Real>> const &v)
        : m_bound_box_(point_type(*v.begin()), point_type(*(v.begin() + 1))) {}

    template <typename V, typename U>
    Box(V const *l, U const *h) : m_bound_box_(box_type({l[0], l[1], l[2]}, {h[0], h[1], h[2]})){};
    Box(box_type const &b) : m_bound_box_(b) {}

   public:
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }
    box_type GetBoundingBox() const override { return m_bound_box_; };

    virtual bool CheckInside(point_type const &x) const override {
        return std::get<0>(m_bound_box_)[0] <= x[0] && x[0] < std::get<1>(m_bound_box_)[0] &&
               std::get<0>(m_bound_box_)[1] <= x[1] && x[1] < std::get<1>(m_bound_box_)[1] &&
               std::get<0>(m_bound_box_)[2] <= x[2] && x[2] < std::get<1>(m_bound_box_)[2];
    }
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
