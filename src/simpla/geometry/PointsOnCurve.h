//
// Created by salmon on 17-11-2.
//

#ifndef SIMPLA_POINTSONCURVE_H
#define SIMPLA_POINTSONCURVE_H

#include "PolyPoints.h"
namespace simpla {
namespace geometry {
struct Curve;
struct PointsOnCurve : public PolyPoints {
    SP_GEO_OBJECT_HEAD(PointsOnCurve, PolyPoints)

   public:
    void SetCurve(std::shared_ptr<const Curve> const&);
    std::shared_ptr<const Curve> GetCurve() const;
    size_type size() const override;
    point_type GetPoint(size_type idx) const override;

    std::vector<Real> const& data() const { return m_data_; }
    std::vector<Real>& data() { return m_data_; }

    void push_back(Real u);

   private:
    std::vector<Real> m_data_;
    std::shared_ptr<const Curve> m_curve_;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POINTSONCURVE_H
