/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <vector>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "geo_object.h"

namespace simpla { namespace geometry
{
/**
 * @ingroup geometry
 * @{
 */

template<int NDIMS> class Polygon;


/**
 *  @brief 2D polygon
 */
template<>
struct Polygon<2> : public Object
{
    using Object::point_type;

    typedef nTuple<Real, 2> point2d_type;

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

public:


    Polygon() { }

    ~Polygon() { }

    Polygon(Polygon const &) = delete;

    std::vector<point2d_type> &data() { return m_polygon_; };

    std::vector<point2d_type> const &data() const { return m_polygon_; };

    void push_back(point_type const &p);

    void deploy();

    virtual int box_intersection(point_type *x0, point_type *x1) const;

    virtual Real nearest_point(point_type *p) const;

    virtual box_type box() const { return box_type(m_x0_, m_x1_); };

    virtual int within(point_type const &x) const;


private:
    point_type m_x0_, m_x1_;
};
/* @} */
}// namespace geometry
}// namespace simpla
#endif //SIMPLA_POLYGON_H
