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
    explicit Line(vector_type const &v);

   public:
    bool IsClosed() const override;
    point_type xyz(Real u) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
