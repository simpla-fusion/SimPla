//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_EDGE_H
#define SIMPLA_EDGE_H
#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
struct gCurve;
struct Edge : public GeoObjectHandle {
    SP_GEO_OBJECT_HEAD(GeoObjectHandle, Edge);

   protected:
    explicit Edge(Axis const &axis, std::shared_ptr<const gCurve> const &curve,
                  std::tuple<Real, Real> const &range = {0, 1});

   public:
    void SetCurve(std::shared_ptr<const gCurve> const &s);
    std::shared_ptr<const gCurve> GetCurve() const;
};
}  // namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_EDGE_H
