/**
 * @file trans_revolve.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_TRANS_REVOLVE_H
#define SIMPLA_TRANS_REVOLVE_H

#include "model_base.h"

#include <vector>

#include "../toolbox/ntuple.h"
#include "../geometry/polygon.h"


namespace simpla
{

template<int ZAXIS = 2>
struct TransformRevolve
{
private:
    static constexpr int XAXIS = (ZAXIS + 1) % 3;
    static constexpr int YAXIS = (ZAXIS + 2) % 3;

//    typedef Polygon<2> polygon_type;
//
//    polygon_type m_polygon_;
//
//    typedef typename polygon_type::point_type point2d_type;

    typedef nTuple<Real, 3> point_type;
    point_type m_x0_{0, 0, 0};
    point_type m_x1_{0, 0, 1};
public:

    TransformRevolve() { };

    ~TransformRevolve() { };

    point_type *apply(point_type *p) const
    {
        return p;
    }

    point_type operator()(point_type const &p) const
    {
        point_type res;
        apply(&res);
        return std::move(res);
    }

    point_type *operator()(point_type *p) const
    {
        return apply(p);
    }
    ////////////////////////////////////////////////////////
    // ModelBase Interface

//    int is_inside(point_type const &x) const
//    {
//        return m_polygon_.is_inside(x[XAXIS], x[YAXIS]);
//    }
//
//    bool box_intersection(point_type *x0, point_type *x1) const
//    {
//        point2d_type y0{(*x0)[XAXIS], (*x0)[YAXIS]};
//        point2d_type y1{(*y0)[XAXIS], (*y0)[YAXIS]};
//
//        bool res = m_polygon_.box_inersection(&y0, &y1);
//
//        x0[XAXIS] = y0[0];
//        x0[XAXIS] = y0[1];
//        x1[YAXIS] = y1[0];
//        x1[YAXIS] = y1[1];
//
//        return res;
//
//    };
//
//    Real nearest_point(point_type *p) const
//    {
//        Real x = (*p)[XAXIS], y = (*p)[YAXIS];
//
//        Real dist = m_polygon_.nearest_point(&x, &y);
//
//        p[XAXIS] = x;
//        p[YAXIS] = y;
//
//        return dist;
//    }
//// ModelBase Interface End
//////////////////////////////////////////////////////////
//
//    template<typename TP>
//    void add_point(TP const &x)
//    {
//        m_polygon_.emplace_back(x[XAXIS], x[YAXIS]);
//    }


};


}//namespace simpla


#endif //SIMPLA_TRANS_REVOLVE_H
