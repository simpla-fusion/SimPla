//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_INTERSECTIONCURVESURFACE_H
#define SIMPLA_INTERSECTIONCURVESURFACE_H

#include <simpla/utilities/Log.h>
#include <memory>
#include <vector>
#include "simpla/SIMPLA_config.h"
namespace simpla {
namespace geometry {
class GeoObject;
class Curve;
struct Intersector {
   protected:
    Intersector();

   private:
    ~Intersector();
    static std::shared_ptr<Intersector> New(std::shared_ptr<const GeoObject> const& geo, Real tolerance = 0.001);
    virtual void Eval(std::shared_ptr<const Curve> const& curve, std::vector<Real>& intersection_point) = 0;
};
template <typename TSurface>
struct IntersectorT : public Intersector {
    IntersectorT(std::shared_ptr<const GeoObject> const& geo, Real tolerance) {}
    ~IntersectorT() {}
    void Eval(std::shared_ptr<const GeoObject> const& curve, std::vector<Real>& intersection_point) const override {
        UNIMPLEMENTED;
    };
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_INTERSECTIONCURVESURFACE_H
