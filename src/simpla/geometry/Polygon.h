/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <vector>

#include <simpla/algebra/nTuple.h>
#include <simpla/utilities/sp_def.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
/**
 * @ingroup geometry
 * @{
 */

template <int NDIMS>
class Polygon;

/**
 *  @brief 2D polygon
 */
template <>
struct Polygon<2> {
    typedef nTuple<Real, 2> point2d_type;

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

   public:
    Polygon() {}

    ~Polygon() {}

    Polygon(Polygon const &) = delete;

    std::vector<point2d_type> &data() { return m_polygon_; };

    std::vector<point2d_type> const &data() const { return m_polygon_; };

    void push_back(point2d_type const &p);

    void deploy();

    virtual Real nearest_point(Real *x, Real *y) const;

    virtual std::tuple<point2d_type, point2d_type> box() const { return std::make_tuple(m_min_, m_max_); };

    virtual int check_inside(Real x, Real y) const;

   private:
    point2d_type m_min_, m_max_;
};
/* @} */
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
