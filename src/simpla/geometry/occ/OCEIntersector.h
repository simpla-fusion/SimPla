//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_OCEINTERSECTOR_H
#define SIMPLA_OCEINTERSECTOR_H

#include "../Intersector.h"
namespace simpla {
namespace geometry {
struct OCEIntersector : public Intersector {
   protected:
    OCEIntersector();
    OCEIntersector(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);
   public:
    ~OCEIntersector();

    static std::shared_ptr<OCEIntersector> New(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);

    size_type GetIntersectionPoints(std::shared_ptr<const Curve> const& curve,
                                    std::vector<Real>& intersection_point) const override;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_OCEINTERSECTOR_H
