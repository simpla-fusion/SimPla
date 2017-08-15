/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include "simpla/SIMPLA_config.h"

#include <vector>

#include "GeoObject.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/data/Data.h"
#include "simpla/engine/SPObject.h"
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
struct Polygon<2> : public GeoObject {
    SP_OBJECT_DECLARE_MEMBERS(Polygon, GeoObject)
   public:
    typedef nTuple<Real, 2> point2d_type;

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

    std::vector<point2d_type> &data() { return m_polygon_; };

    std::vector<point2d_type> const &data() const { return m_polygon_; };

    void push_back(point2d_type const &p);

    void deploy();

    Real nearest_point(Real *x, Real *y) const;
    bool check_inside(Real x, Real y) const;

    std::tuple<point_type, point_type> BoundingBox() const override {
        return std::move(std::make_tuple(point_type{m_min_[0], m_min_[1], 0}, point_type{m_max_[0], m_max_[1], 1}));
    };

   private:
    point2d_type m_min_, m_max_;
};
/* @} */
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
