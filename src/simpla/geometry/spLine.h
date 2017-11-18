//
// Created by salmon on 17-10-21.
//

#ifndef SIMPLA_LINE_H
#define SIMPLA_LINE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
#include "Curve.h"
namespace simpla {
namespace geometry {
struct spLine : public Shape {
    SP_SHAPE_HEAD(Shape, spLine, Line)

    explicit spLine(point_type const &p0, point_type const &p1);
    explicit spLine(point_type const &p0, vector_type const &dir, Real length);

    point_type GetStartPoint() const;
    point_type GetEndPoint() const;
    vector_type GetDirection() const;
    Real GetLength() const { return m_length_; }
    void SetLength(Real l) { m_length_ = l; }

   private:
    Real m_length_ = 1;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
