//
// Created by salmon on 17-11-7.
//

#ifndef SIMPLA_WEDGE_H
#define SIMPLA_WEDGE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "Shape.h"
namespace simpla {
namespace geometry {

struct spWedge : public Shape {
    SP_SHAPE_HEAD(Shape, spWedge, Wedge)
    explicit spWedge(vector_type const &v, Real ltx);
    point_type xyz(Real u, Real v, Real w) const;
    vector_type GetExtents() const { return m_extents_; }
    void SetExtents(vector_type const &extents) { m_extents_ = extents; }
    Real GetLTX() const { return m_ltx_; }
    void SetLTX(Real l) { m_ltx_ = l; }

   protected:
    vector_type m_extents_{1, 1, 1};
    Real m_ltx_ = PI / 2;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_WEDGE_H
