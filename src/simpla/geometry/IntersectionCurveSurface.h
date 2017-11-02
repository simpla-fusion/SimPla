//
// Created by salmon on 17-11-2.
//

#ifndef SIMPLA_ITERSECTIONCURVESURFACE_H
#define SIMPLA_ITERSECTIONCURVESURFACE_H

#include <simpla/utilities/Factory.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
struct Curve;
struct Surface;
class IntersectionCurveSurface : public Factory<IntersectionCurveSurface> {
    FACTORY_HEAD(IntersectionCurveSurface)
   public:
    static std::shared_ptr<this_type> New(std::string const &key = "");
    template <typename... Args>
    static std::shared_ptr<this_type> New(std::shared_ptr<const Surface> const &g, Real tolerance,
                                          std::string const &key = "") {
        auto res = New(key);
        res->SetUp(g, tolerance);
        return res;
    }
    virtual void SetUp(std::shared_ptr<const Surface> const &g, Real tolerance);
    virtual void TearDown();

    virtual size_type Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *u) const;

   protected:
    std::shared_ptr<const Surface> m_surface_ = nullptr;
    Real m_tolerance_ = SP_GEO_DEFAULT_TOLERANCE;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_ITERSECTIONCURVESURFACE_H
