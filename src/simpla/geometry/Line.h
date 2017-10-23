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
    Line() = default;
    Line(Line const &) = default;
    explicit Line(std::shared_ptr<Axis> const &axis) : Curve(axis){};
    Line(point_type const &p0, point_type const &p1) : Curve(Axis::New(p0, p1 - p0)){};

   public:
    ~Line() override = default;

    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; }

    point_type Value(Real u) const override { return m_axis_->Coordinates(u); }
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
