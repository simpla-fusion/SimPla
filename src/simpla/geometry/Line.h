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
struct Line : public Curve {
    SP_GEO_OBJECT_HEAD(Line, Curve);

   protected:
    explicit Line(point_type const &p0, point_type const &p1);
    explicit Line(point_type const &p0, vector_type const &dir, Real length);

   public:
    bool IsClosed() const override;
    point_type xyz(Real u) const override;
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
