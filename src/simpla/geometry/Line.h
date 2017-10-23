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
    template <typename... Args>
    Line(Args &&... args) : Curve(std::forward<Args>(args)...){};

   public:
    ~Line() override = default;

    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; }

    point_type Value(Real u) const override { return m_axis_.o + u * m_axis_.x; }
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
