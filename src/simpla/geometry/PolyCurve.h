//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_POLYCURVE_H
#define SIMPLA_POLYCURVE_H

#include "Curve.h"
namespace simpla {
namespace geometry {

struct PolyCurve : public Curve {
    SP_GEO_OBJECT_HEAD(PolyCurve, Curve);

   protected:
    PolyCurve();
    PolyCurve(PolyCurve const &);
    explicit PolyCurve(Axis const &axis) : PolyCurve() { Curve::SetAxis(axis); }

   public:
    ~PolyCurve() override;
    bool IsClosed() const override;
    void PushBack(std::shared_ptr<Curve> const &, Real length = SP_SNaN);
    void PushFront(std::shared_ptr<Curve> const &, Real length = SP_SNaN);
    void Foreach(std::function<void(std::shared_ptr<Curve> const &)> const &);
    void Foreach(std::function<void(std::shared_ptr<const Curve> const &)> const &) const;
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYCURVE_H
