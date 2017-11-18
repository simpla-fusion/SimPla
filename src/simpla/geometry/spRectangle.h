//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_RECTANGLE_H
#define SIMPLA_RECTANGLE_H

#include <simpla/SIMPLA_config.h>
#include "Shape.h"
namespace simpla {
namespace geometry {
struct spRectangle : public Shape {
    SP_SERIALIZABLE_HEAD(Shape, spRectangle);

   protected:
    spRectangle(Real l, Real w);
    spRectangle(Axis const& axis, Real l, Real w);

   public:
    //    point_type xy(Real u, Real v) const override { return m_axis_.xyz(u * m_l_, v * m_w_, 0); };
    //    point_type uv(Real x, Real y) const override { return point_type{x / m_l_, y / m_w_, 0}; };

    Real GetWidth() const { return m_w_; }
    void SetWidth(Real w) { m_w_ = w; }
    Real GetLength() const { return m_l_; }
    void SetLength(Real h) { m_l_ = h; }

   protected:
    Real m_l_ = 1.0, m_w_ = 1.0;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_RECTANGLE_H
