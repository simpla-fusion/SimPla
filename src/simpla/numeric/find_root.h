/**
 *  @file find_root.h
 *
 * @date 2014-7-6
 * @author salmon
 */

#ifndef FIND_ROOT_H_
#define FIND_ROOT_H_

#include <tuple>
#include "../toolbox/nTuple.h"

namespace simpla
{
//!  @ingroup numeric
//! @{

/**
 *
 * using Newton's method find root
 *
 * @param x0 \f$x_{0}\f$
 * @param x1 \f$x_{1}\f$
 * @param f function
 * @param y_m \f$y_m\f$
 * @param resolution
 * @param max_iterator_num
 * @return
 */
template<typename TX, typename TY>
bool find_root(std::function<TY(TX const &)> const &f, TY const &y_m, TX x0, TX *xio,
               double resolution = 0.001, size_t max_iterator_num = 10000)
{
    auto x1 = *xio;
    // @todo need change to Newton method!!!
    resolution *= inner_product(x1 - x0, x1 - x0);

    bool success = false;

    TX x;

    TY y0 = f(x0);
    TY y1 = f(x1);
    size_t count = 0;
    do
    {
        ++count;
        TX x = (x0 + x1) / 2;

        TY y = f(x);

        if (y < y_m == y0 < y_m)
        {
            y0 = y;
            x0 = x;
        }
        else if (y < y_m == y1 < y_m)
        {
            y1 = y;
            x1 = x;
        }

        success = (y0 < y_m) != (y1 < y_m);

        if (!success)
            break;

    } while (inner_product(x1 - x0, x1 - x0) > resolution
             && count < max_iterator_num);
    *xio = x0;

    return success && (count < max_iterator_num);

}

/**
 *
 * using mid-point(binary search) method find root
 *
 * @param x0 \f$x_{0}\f$
 * @param x1 \f$x_{1}\f$
 * @param f function
 * @param resolution
 * @param max_iterator_num
 * @return
 */
template<typename TX>
bool find_root(std::function<bool(TX const &)> const &f, TX x0, TX *xio, double resolution = 0.001,
               size_t max_iterator_num = 10000)
{
    auto x1 = *xio;
    resolution *= std::abs(x1 - x0);

    bool success = false;

    bool y0 = f(x0);
    bool y1 = f(x1);
    size_t count = 0;
    do
    {
        ++count;
        TX x = (x0 + x1) / 2;

        bool y = f(x);

        if (y == y0)
        {
            y0 = y;
            x0 = x;
        }
        else if (y == y1)
        {
            y1 = y;
            x1 = x;
        }

        success = y0 != y1;

        if (!success)
            break;

    } while (std::abs(x1 - x0) > resolution && count < max_iterator_num);

    *xio = x0;
    return success && (count < max_iterator_num);
}

//!  @}
}// namespace simpla

#endif /* FIND_ROOT_H_ */
