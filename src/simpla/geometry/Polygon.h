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

    bool IsClosed() const override;

    std::vector<point2d_type> &data();
    std::vector<point2d_type> const &data() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
