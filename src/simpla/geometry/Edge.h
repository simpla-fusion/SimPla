//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_EDGE_H
#define SIMPLA_EDGE_H
#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
struct Curve;
struct Edge : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Edge);

   protected:
    Edge(std::shared_ptr<const Curve> const &surface, Real l, Real w);
    Edge(std::shared_ptr<const Curve> const &surface, std::tuple<Real, Real> const &range);

   public:
    void SetCurve(std::shared_ptr<const Curve> const &s) { m_curve_ = s; }
    std::shared_ptr<const Curve> GetCurve() const { return m_curve_; }
    void SetParameterRange(Real umin, Real umax) const { m_range_ = std::tie(umin, umax); };
    std::tuple<Real, Real> const &GetParameterRange() const { return m_range_; };

   private:
    std::shared_ptr<const Curve> m_curve_;
    std::tuple<Real, Real> m_range_{0, 1};
};
}  // namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_EDGE_H
