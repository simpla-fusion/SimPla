//
// Created by salmon on 17-10-27.
//

#ifndef SIMPLA_POLYLINE2D_H
#define SIMPLA_POLYLINE2D_H

#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Polyline2d : public Curve {
    SP_GEO_OBJECT_HEAD(Polyline2d, Curve);

   protected:
    Polyline2d();
    Polyline2d(Polyline2d const &);
    explicit Polyline2d(Axis const &axis);

   public:
    ~Polyline2d() override;

    bool IsClosed() const override;
    bool IsPeriodic() const override;

    Real GetPeriod() const override;
    Real GetMinParameter() const override;
    Real GetMaxParameter() const override;

    point_type Value(Real u) const override;
    void Close();
    void AddUV(Real u, Real v);                                                       // add xyz
    void AddPoint(point_type const &xyz, Real tolerance = SP_GEO_DEFAULT_TOLERANCE);  // add xyz
    point_type StartPoint(int n) const;
    point_type EndPoint(int n) const;
    size_type size() const;
    std::vector<nTuple<Real, 2>> &data();
    std::vector<nTuple<Real, 2>> const &data() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYLINE2D_H
