/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <vector>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Polygon : public Curve {
    SP_GEO_OBJECT_HEAD(Polygon, Curve)

   public:
    void Open();
    void Close();
    bool IsClosed() const override;

    point_type xyz(Real u) const override;

    void Add(Real u, Real v);
    void Add(Real u, Real v, Real w);
    void Add(size_type num, Real const *u, Real const *v, Real const *w = nullptr);

    std::vector<point2d_type> &data();
    std::vector<point2d_type> const &data() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
