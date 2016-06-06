/**
 * @file GeneralMap.h
 * @author salmon
 * @date 2015-10-29.
 */

#ifndef SIMPLA_MAP_GENERAL_H
#define SIMPLA_MAP_GENERAL_H

#include "../../gtl/primitives.h"
#include "LinearMap.h"

namespace simpla { namespace mesh
{

/**
 *   \f$ \left( f \circ g \right)\left( x \right)=f\left( g\left( x \right) \right)  \f$
 */
template<typename LEFT>
struct GeneralMap
{
private:

    typedef GeneralMap<LEFT> this_type;

    typedef LEFT left_action;
    typedef LinearMap right_action;


    left_action m_left_;
    right_action m_right_;
public:


    GeneralMap() { }

    virtual ~GeneralMap() { }

    void swap(this_type &other)
    {
        m_left_.swap(other.m_left_);
        m_right_.swap(other.m_right_);
    }

    template<typename TB0, typename TB1, typename TD, typename ...Others>
    void set(TB0 const &src_box, TB1 const &dest_box, TD const &dims, Others &&...others)
    {
        m_left_.set(std::forward<Others>(others)...);

        m_right_.set(src_box,
                     std::make_tuple(m_left_.inv_map(std::get<0>(dest_box)),
                                     m_left_.inv_map(std::get<1>(dest_box))));
    }

    template<typename TP>
    TP *map(TP *x) const
    {
        return m_left_.map(m_right_.map(x));
    }

    template<typename TP>
    TP *inv_map(TP *y) const
    {
        return m_right_.inv_map(m_left_.inv_map(y));
    }

    template<typename TP>
    TP map(TP const &x) const
    {
        return std::move(m_left_.map(m_right_.map(x)));
    }

    template<typename TP>
    TP inv_map(TP const &y) const
    {
        return std::move(m_right_.inv_map(m_left_.inv_map(y)));
    }


}; // General Map

template<typename TP=nTuple<Real, 3>, int N = 0>
struct SquareMap
{
private:

    typedef SquareMap<TP, N> this_type;

    typedef TP point_type;
public:


    SquareMap() { }

    virtual ~SquareMap() { }

    void swap(this_type &other) { }

    template<typename ...Others> void set(Others &&...others) { }


    point_type *map(point_type *x) const
    {
        (*x)[N] = (*x)[N] * (*x)[N];
        return x;
    }


    point_type *inv_map(point_type *x) const
    {
        (*x)[N] = std::sqrt((*x)[N]);
        return x;


    }

    template<typename U>
    point_type map(U const &x) const
    {
        point_type res;
        res = x;
        map(&res);

        return std::move(res);
    }

    template<typename U>
    point_type inv_map(U const &y) const
    {
        point_type res;
        res = y;
        inv_map(&res);
        return std::move(res);
    }
};
}//namespace get_mesh
}//namespace simpla
#endif //SIMPLA_MAP_GENERAL_H
