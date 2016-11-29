/**
 * @file polygon.cpp
 * @author salmon
 * @date 2015-11-18.
 */
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/nTupleExt.h>
#include "Polygon.h"
#include "GeoAlgorithm.h"

namespace simpla { namespace geometry
{

//int Polygon<2>::box_intersection(point_type *x0, point_type *x1) const
//{
//    return geometry::box_intersection(m_min_, m_max_, x0, x1);
//}


Real Polygon<2>::nearest_point(Real *x, Real *y) const
{

    typedef nTuple<Real, 2> Vec2;

    point2d_type x0;

    x0[0] = *x;
    x0[1] = *y;

    point2d_type p0 = m_polygon_.back();

    point2d_type p1;

    auto it = m_polygon_.begin();

    Real d2 = std::numeric_limits<Real>::max();


    while (it != m_polygon_.end())
    {

        p1 = *it;
        ++it;

        Vec2 u, v;

        u = x0 - p0;
        v = p1 - p0;

        Real v2 = inner_product(v, v);

        auto s = inner_product(u, v) / v2;


        point2d_type p;

        if (s < 0) { s = 0; }
        else if (s > 1) { s = 1; }

        p = ((1 - s) * p0 + s * p1);
        /**
         * if \f$ v \times u \cdot e_z >0 \f$ then `in` else `out`
         */

        Real dd = inner_product(x0 - p, x0 - p);


        if (std::abs(dd) < std::abs(d2))
        {
            d2 = dd;
            (*x) = x0[0] - u[0];
            (*y) = x0[1] - u[1];

        }
        p0 = p1;

    }

    d2 = std::sqrt(d2);

    return check_inside(*x, *y) > 0 ? d2 : -d2;
}


void Polygon<2>::push_back(point2d_type const &pp) { m_polygon_.push_back(pp); }

void Polygon<2>::deploy()
{
    size_t num_of_vertex_ = m_polygon_.size();
    constant_.resize(num_of_vertex_);
    multiple_.resize(num_of_vertex_);

    for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
    {
        if (m_polygon_[j][1] == m_polygon_[i][1])
        {
            constant_[i] = m_polygon_[i][0];
            multiple_[i] = 0;
        } else
        {
            constant_[i] = m_polygon_[i][0]
                           - (m_polygon_[i][1] * m_polygon_[j][0])
                             / (m_polygon_[j][1] - m_polygon_[i][1])
                           + (m_polygon_[i][1] * m_polygon_[i][0])
                             / (m_polygon_[j][1] - m_polygon_[i][1]);
            multiple_[i] = (m_polygon_[j][0] - m_polygon_[i][0])
                           / (m_polygon_[j][1] - m_polygon_[i][1]);
        }
        j = i;
    }

    m_min_ = m_polygon_.front();
    m_max_ = m_polygon_.front();

    for (auto const &p:m_polygon_) { geometry::extent_box(&m_min_, &m_max_, &p[0]); }
}


int Polygon<2>::check_inside(Real x, Real y) const
{
     if ((x >= m_min_[0]) && (y >= m_min_[1]) && (x < m_max_[0]) && (y < m_max_[1]))
    {
        size_t num_of_vertex_ = m_polygon_.size();

        bool oddNodes = false;

        for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
        {
            if (((m_polygon_[i][1] < y) && (m_polygon_[j][1] >= y))
                || ((m_polygon_[j][1] < y) && (m_polygon_[i][1] >= y)))
            {
                oddNodes ^= (y * multiple_[i] + constant_[i] < x);
            }

            j = i;
        }

        return oddNodes ? 1 : 0;
    } else { return 0; }
}

}}//namespace simpla//namespace  geometry