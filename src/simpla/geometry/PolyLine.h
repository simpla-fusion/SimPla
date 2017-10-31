//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_POLYLINE_H
#define SIMPLA_POLYLINE_H

#include "Curve.h"
namespace simpla {
namespace geometry {

struct PolyLine : public Curve {
    SP_GEO_OBJECT_HEAD(PolyLine, Curve);

   protected:
    PolyLine();
    PolyLine(PolyLine const &);
    explicit PolyLine(Axis const &axis);

   public:
    ~PolyLine() override;
    bool IsClosed() const override;

    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYLINE_H
