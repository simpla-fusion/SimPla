/**
 *  @file rectangle_distribution.h
 *
 *  created on: 2013-10-24
 *      Author: salmon
 */

#ifndef RECTANGLE_DISTRIBUTION_H_
#define RECTANGLE_DISTRIBUTION_H_

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include "../toolbox/nTuple.h"
#include "../sp_def.h"

namespace simpla
{
/** @ingroup numeric */

template<int NDIMS>
class rectangle_distribution
{

public:

    typedef nTuple <Real, NDIMS> value_type;

    rectangle_distribution()
    {
        nTuple<double, NDIMS> xmin, xmax;

        xmin = 0;
        xmax = 1;

        reset(xmin, xmax);
    }

    rectangle_distribution(rectangle_distribution const &other) :
            xmax_(other.xmax_), xmin_(other.xmin_), l_(other.l_)
    {

    }

    template<typename TBox>
    rectangle_distribution(TBox const &b)
    {
        reset(traits::get<0>(b), traits::get<1>(b));
    }

    rectangle_distribution(nTuple<double, NDIMS> const &xmin,
                           nTuple<double, NDIMS> const &xmax)
    {
        reset(xmin, xmax);
    }


    ~rectangle_distribution()
    {
    }


    inline void reset(nTuple<double, NDIMS> const &xmin,
                      nTuple<double, NDIMS> const &xmax)
    {
        xmin_ = xmin;
        xmax_ = xmax;

        for (int i = 0; i < NDIMS; ++i)
        {

            if (xmax_[i] > xmin_[i])
            {
                l_[i] = (xmax_[i] - xmin_[i]);
            }
            else
            {
                l_[i] = 0;
            }
        }
    }

    template<typename Generator>
    nTuple<double, NDIMS> operator()(Generator &g) const
    {
        nTuple<double, NDIMS> res;

        for (int i = 0; i < NDIMS; ++i)
        {
            res[i] = static_cast<double>(g() - g.min())
                     / static_cast<double>(g.max() - g.min()) * l_[i] + xmin_[i];
        }

        return std::move(res);

    }

    template<typename Generator, typename T>
    void operator()(Generator &g, T *res) const
    {

        for (int i = 0; i < NDIMS; ++i)
        {
            res[i] = static_cast<double>(g() - g.min())
                     / static_cast<double>(g.max() - g.min()) * l_[i] + xmin_[i];
        }
    }

private:
    nTuple<double, NDIMS> xmin_;
    nTuple<double, NDIMS> xmax_;
    nTuple<double, NDIMS> l_;

};

}  // namespace simpla

#endif /* RECTANGLE_DISTRIBUTION_H_ */
