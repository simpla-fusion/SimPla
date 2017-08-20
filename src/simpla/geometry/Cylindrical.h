//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include "simpla/SIMPLA_config.h"

#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Cylindrical : public GeoObject {
    SP_OBJECT_HEAD(Cylindrical, GeoObject)
    Real m_radius_ = 1;
    point_type m_axe0_{0, 0, 0};
    point_type m_axe1_{0, 0, 1};

   protected:
    Cylindrical(Real r, point_type o0, point_type o1) : m_radius_(r), m_axe0_(std::move(o0)), m_axe1_(std::move(o1)) {}

   public:
    box_type BoundingBox() const override {
        box_type b;
        std::get<0>(b) = m_axe0_ - m_radius_;
        std::get<1>(b) = m_axe1_ + m_radius_;
        return std::move(b);
    };

    bool CheckInside(point_type const &x) const override {
        return dot((x - m_axe0_), (x - m_axe0_)) - m_radius_ * m_radius_ < 0;
    }
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
