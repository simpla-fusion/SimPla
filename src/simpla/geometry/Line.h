//
// Created by salmon on 17-10-21.
//

#ifndef SIMPLA_LINE_H
#define SIMPLA_LINE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
#include "Curve.h"
#include "ParametricCurve.h"
namespace simpla {
namespace geometry {
struct Line : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Line, ParametricCurve);

   protected:
    Line();
    Line(Line const &);
    explicit Line(Axis const &axis, Real alpha0 = -SP_INFINITY, Real alpha1 = SP_INFINITY);
    explicit Line(point_type const &p0, point_type const &p1);
    explicit Line(vector_type const &v);

   public:
    ~Line() override;

    bool IsClosed() const override;

    point_type xyz(Real u) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
