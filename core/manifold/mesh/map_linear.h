/**
 * @file map_linear.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_LINEAR_MAP_H
#define SIMPLA_LINEAR_MAP_H


#include "../../gtl/primitives.h"

namespace simpla { namespace mesh
{


struct LinearMap
{
private:

    typedef LinearMap this_type;

    typedef nTuple<Real, 3> point_type;


    point_type m_map_orig_ = {0, 0, 0};

    point_type m_map_scale_ = {1, 1, 1};

    point_type m_inv_map_orig_ = {0, 0, 0};

    point_type m_inv_map_scale_ = {1, 1, 1};

public:

    LinearMap() { }

    virtual   ~LinearMap() { }

    void swap(this_type &other)
    {
        std::swap(m_inv_map_orig_, other.m_inv_map_orig_);
        std::swap(m_map_orig_, other.m_map_orig_);
        std::swap(m_map_scale_, other.m_map_scale_);
        std::swap(m_inv_map_scale_, other.m_inv_map_scale_);
    }

    template<typename TB0, typename TB1>
    void set(TB0 const &src_box, TB1 const &dest_box, nTuple<size_t, 3> const &dims = {10, 10, 10})
    {

        point_type src_min_, src_max_;

        point_type dest_min, dest_max;

        std::tie(src_min_, src_max_) = src_box;

        std::tie(dest_min, dest_max) = dest_box;

        for (int i = 0; i < 3; ++i)
        {
            m_map_scale_[i] = (dest_max[i] - dest_min[i]) / (src_max_[i] - src_min_[i]);

            m_inv_map_scale_[i] = (src_max_[i] - src_min_[i]) / (dest_max[i] - dest_min[i]);


            m_map_orig_[i] = dest_min[i] - src_min_[i] * m_map_scale_[i];

            m_inv_map_orig_[i] = src_min_[i] - dest_min[i] * m_inv_map_scale_[i];

        }
    }


    point_type inv_map(point_type const &x) const
    {

        point_type res;


        res[0] = std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);

        res[1] = std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);

        res[2] = std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);

        return std::move(res);
    }

    point_type map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);

        res[1] = std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);

        res[2] = std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);

        return std::move(res);
    }

};

}//namespace  mesh
}//namespace simpla

#endif //SIMPLA_LINEAR_MAP_H
