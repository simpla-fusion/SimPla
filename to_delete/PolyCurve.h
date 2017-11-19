//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_POLYCURVE_H
#define SIMPLA_POLYCURVE_H

#include "Curve.h"
namespace simpla {
namespace geometry {

struct PolyCurve : public Curve {
    SP_GEO_OBJECT_HEAD(Curve, PolyCurve);

   public:
    point_type xyz(Real u) const override;

    bool IsClosed() const override;
    void PushBack(std::shared_ptr<Curve> const &, Real length = SP_SNaN);
    void PushFront(std::shared_ptr<Curve> const &, Real length = SP_SNaN);
    void Foreach(std::function<void(std::shared_ptr<Curve> const &)> const &);
    void Foreach(std::function<void(std::shared_ptr<const Curve> const &)> const &) const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYCURVE_H
