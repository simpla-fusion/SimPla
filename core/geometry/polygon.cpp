/**
 * @file polygon.cpp
 * @author salmon
 * @date 2015-11-18.
 */
#include "../gtl/ntuple_ext.h"
#include "polygon.h"
#include "geo_algorithm.h"

namespace simpla { namespace geometry
{

int Polygon<2>::box_intersection(point_type *x0, point_type *x1) const
{
    return geometry::box_intersection(m_x0_, m_x1_, x0, x1);
}

Real  Polygon<2>::nearest_point(point_type *p) const
{
    return geometry::nearest_point_to_polygon(m_polygon_.begin(), m_polygon_.end(), p);
}

Real Polygon<2>::normals(point_type *x, vector_type *v) const
{
    //UNIMPLEMENTED;
    return 0;
}


void Polygon<2>::push_back(point_type const &pp)
{
    point2d_type p;
    p = pp;
    m_polygon_.push_back(p);
}

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
        }
        else
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

    m_x0_ = m_polygon_.front();
    m_x1_ = m_polygon_.front();

    for (auto const &p:m_polygon_)
    {
        geometry::extent_box(&m_x0_, &m_x1_, p);
    }

    m_x0_[2] = std::numeric_limits<Real>::min();
    m_x1_[2] = std::numeric_limits<Real>::max();

}


int  Polygon<2>::within(point_type const &p) const
{
    Real x = p[0];
    Real y = p[1];
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
}

}//namespace  geometry
}//namespace simpla