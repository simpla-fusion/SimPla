//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_OCEINTERSECTOR_H
#define SIMPLA_OCEINTERSECTOR_H

#include "../Intersector.h"
namespace simpla {
namespace geometry {
struct IntersectorOCE : public Intersector {
    SP_OBJECT_HEAD(IntersectorOCE, Intersector)
   protected:
    IntersectorOCE(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);

   public:
    static std::shared_ptr<IntersectorOCE> New(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);

    size_type GetIntersectionPoints(std::shared_ptr<const GeoObject> const& curve,
                                    std::vector<Real>& intersection_point) const override;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_OCEINTERSECTOR_H
