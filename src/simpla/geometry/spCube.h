//
// Created by salmon on 17-11-18.
//

#ifndef SIMPLA_SPCUBE_H
#define SIMPLA_SPCUBE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/Configurable.h>
#include "Shape.h"
namespace simpla {
namespace geometry {

struct spCube : public Shape {
    SP_SERIALIZABLE_HEAD(Shape, Cube)

   protected:
    explicit spCube(box_type const &b) : m_min_(std::get<0>(b)), m_max_(std::get<1>(b)) {}

   public:
    point_type xyz(Real u, Real v, Real w) const override { return point_type{u, v, w} * (m_max_ - m_min_) + m_min_; };
    point_type uvw(Real x, Real y, Real z) const override {
        return (point_type{x, y, z} - m_min_) / (m_max_ - m_min_);
    };

    void SetBox(box_type const &b) { std::tie(m_min_, m_max_) = b; };
    box_type const &GetBox() const { return std::make_tuple(m_min_, m_max_); };

   private:
    point_type m_min_{0, 0, 0};
    point_type m_max_{1, 1, 1};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SPCUBE_H
