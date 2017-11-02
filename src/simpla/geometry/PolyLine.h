//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_POLYLINE_H
#define SIMPLA_POLYLINE_H

#include "BoundedCurve.h"
namespace simpla {
namespace geometry {

struct PolyLine : public BoundedCurve {
    SP_GEO_OBJECT_HEAD(PolyLine, BoundedCurve);

   protected:
    PolyLine();
    PolyLine(PolyLine const &);
    explicit PolyLine(Axis const &axis);

   public:
    ~PolyLine() override;
    bool IsClosed() const override;
    point_type xyz(Real u) const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYLINE_H
