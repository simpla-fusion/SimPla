/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <simpla/SIMPLA_config.h>
#include <vector>
#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Polygon : public Surface {
    SP_GEO_OBJECT_HEAD(Polygon, Surface)

   protected:
    Polygon();
    Polygon(Polygon const &);
    explicit Polygon(Axis const &axis);

   public:
    ~Polygon() override;
    void Open();
    void Close();
    void push_back(Real u, Real v);
    void push_back(size_type num, Real const *u, Real const *v);
    void push_back(nTuple<Real, 2> const &p) { push_back(p[0], p[1]); }
    std::tuple<bool, bool> IsClosed() const override;
    std::tuple<bool, bool> IsPeriodic() const override;
    nTuple<Real, 2> GetPeriod() const override;
    nTuple<Real, 2> GetMinParameter() const override;
    nTuple<Real, 2> GetMaxParameter() const override;

    int CheckOverlap(box_type const &) const override;
    int FindIntersection(std::shared_ptr<const GeoObject> const &, std::vector<Real> &, Real tolerance) const override;
    point_type Value(Real u, Real v) const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
