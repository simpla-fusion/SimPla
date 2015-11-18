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
#include "../numeric/find_root.h"
#include "../gtl/utilities/log.h"
#include "geo_object.h"
#include "geo_algorithm.h"

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
    std::vector<nTuple<Real, 2>> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;
public:

    typedef nTuple<Real, 2> point_type;

    Polygon()
    {
    }

    ~Polygon()
    {
    }

    std::vector<nTuple<Real, 2>> &data()
    {
        return m_polygon_;
    };

    void push_back(Real x, Real y)
    {
        m_polygon_.emplace_back(x, y);

        if (m_polygon_.empty())
        {
            m_x0_ = m_polygon_.back();
            m_x1_ = m_x0_;
        }
        else
        {
            geometry::extent_box(m_polygon_.back(), &m_x0_, &m_x1_);

        }

    }

    void deploy()
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

    }

    bool box_intersection(point_type *x0, point_type *x1) const
    {
        return geometry::box_intersection(m_x0_, m_x1_, x0, x1);
    }

    Real nearest_point(point_type *p) const
    {

        return geometry::nearest_point_to_polygon(m_polygon_.begin(), m_polygon_.end(), p);
    }


    inline int is_inside(point_type const &p) const
    {
        return geometry::in_box(p, m_x0_, m_x1_) && is_inside(p[0], p[1]);
    }

private:
    inline int is_inside(Real x, Real y) const
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
    }

public:

    template<typename TP>
    bool intersection(TP const &x0, TP const &x1, double error = 0.001) const
    {
        std::function<double(TP const &)> fun = [this](TP const &x) { return is_inside(x); };

        return find_root(fun, error, x0, x1);
    }

private:
    point_type m_x0_, m_x1_;
};

}// namespace geometry
}// namespace simpla
#endif //SIMPLA_POLYGON_H
