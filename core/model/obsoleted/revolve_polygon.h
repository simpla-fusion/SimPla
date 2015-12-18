/**
 * @file revolve_polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_REVOLVE_POLYGON_H
#define SIMPLA_REVOLVE_POLYGON_H

#include "../geometry/polygon.h"
#include "model_base.h"

namespace simpla
{

template<int ZAXIS>
struct RevolvePolygon : public ModelBase
{
    using ModelBase::point_type;
    using ModelBase::box_type;

    static constexpr int XAXIS = (ZAXIS + 1) % 3;
    static constexpr int YAXIS = (ZAXIS + 2) % 3;

    void add_point(point_type const &p)
    {
        m_polygon_.emplace_back(p[XAXIS], p[YAXIS]);
    }

    void deploy()
    {
        m_polygon_.deploy();
    }

    box_type box() const
    {
        typename polygon_type::point_type x0, x1;
        std::tie(x0, x1) = m_polygon_.box();
        box_type b;
        std::get<0>(b)[XAXIS] = x0[0];
        std::get<0>(b)[YAXIS] = x0[1];
        std::get<1>(b)[XAXIS] = x1[0];
        std::get<1>(b)[YAXIS] = x1[1];
    }

    virtual int is_inside(point_type const &p) const
    {
        return m_polygon_.is_inside(p[XAXIS], p[YAXIS]);
    };

    virtual int box_intersection(point_type *x0, point_type *x1) const
    {
        typename polygon_type::point_type y0{(*x0)[XAXIS], (*x0)[YAXIS]};
        typename polygon_type::point_type y1{(*x1)[XAXIS], (*x1)[YAXIS]};

        auto res = m_polygon_.box_intersection(&y0, &y1);

        (*x0)[XAXIS] = y0[0];
        (*x0)[YAXIS] = y0[1];
        (*x1)[XAXIS] = y1[0];
        (*x1)[XAXIS] = y1[1];

        return res;

    };

    virtual Real nearest_point(point_type *p) const
    {
        return m_polygon_.nearest_point(&((*p)[XAXIS]), &((*p)[YAXIS]));
    };

private:
    typedef Polygon<2> polygon_type;
    polygon_type m_polygon_;
};
}//namespace simpla
#endif //SIMPLA_REVOLVE_POLYGON_H
