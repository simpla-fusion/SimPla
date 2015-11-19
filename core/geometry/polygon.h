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
template<int NDIMS = 2> class Polygon;

/**
 *  @ingroup numeric  geometry_algorithm
 *  \brief check a point in 2D polygon
 */
template<>
class Polygon<2> : public Object
{
    using Object::point_type;
    typedef nTuple<Real, 3> point2d_type;

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

public:


    Polygon() { }

    ~Polygon() { }

    Polygon(Polygon const &) = delete;

    std::vector<point2d_type> &data() { return m_polygon_; };

    std::vector<point2d_type> const &data() const { return m_polygon_; };

    void push_back(Real x, Real y);

    void deploy();

    int box_intersection(point_type *x0, point_type *x1) const;

    Real nearest_point(point_type *p) const;

private:
    int within(Real x, Real y) const;

public:
    box_type box() const
    {
        return box_type(m_x0_, m_x1_);
    };

    /**
     * @return  \f$ (x,y,z) \is_inside M\f$ ? 1 : 0
     */
    virtual int within(point_type const &x) const
    {
        return within(x[0], x[1]);
    };

    bool intersection(point_type const &x0, point_type const &x1, double error = 0.001) const;
//    {
//        std::function<double(TP const &)> fun = [this](TP const &x) { return within(x); };
//
//        return find_root(fun, error, x0, x1);
//    }

private:
    point_type m_x0_, m_x1_;
};

}// namespace geometry
}// namespace simpla
#endif //SIMPLA_POLYGON_H
