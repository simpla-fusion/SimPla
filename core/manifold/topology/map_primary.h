/**
 * @file map_primary.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_MAP_PRIMARY_H
#define SIMPLA_MAP_PRIMARY_H

#include "../../gtl/primitives.h"

namespace simpla
{
struct IdentifyMap
{
    template<typename T>
    constexpr T const &map(T const &v) const { return v; }

    template<typename T>
    constexpr T const &inv_map(T const &v) const { return v; }
};

struct LinearMap
{

private:

    typedef LinearMap this_type;


    Real m_inv_map_orig_ = 0;

    Real m_map_orig_ = 0;

    Real m_map_scale_ = 1;

    Real m_inv_map_scale_ = 1;

public:

    LinearMap() { }

    ~LinearMap() { }

    void swap(this_type &other)
    {

        std::swap(m_map_orig_, other.m_map_orig_);
        std::swap(m_map_scale_, other.m_map_scale_);

        std::swap(m_inv_map_orig_, other.m_inv_map_orig_);
        std::swap(m_inv_map_scale_, other.m_inv_map_scale_);
    }

    void set(Real src_min, Real src_max, Real dest_min, Real dest_max, bool is_zero_axe = false)
    {

        if (is_zero_axe)
        {

            m_map_scale_ = 0;

            m_inv_map_scale_ = 0;

            m_map_orig_ = dest_min;

            m_inv_map_orig_ = src_min;

        } else
        {

            m_map_scale_ = (dest_max - dest_min) / (src_max - src_min);

            m_inv_map_scale_ = (src_max - src_min) / (dest_max - dest_min);


            m_map_orig_ = dest_min - src_min * m_map_scale_;

            m_inv_map_orig_ = src_min - dest_min * m_inv_map_scale_;

        }

    }


    Real map(Real x) const
    {

        return std::fma(x, m_map_scale_, m_map_orig_);
    }

    Real inv_map(Real x) const
    {

        return std::fma(x, m_inv_map_scale_, m_inv_map_orig_);
    }
};
}//namespace simpla

#endif //SIMPLA_MAP_PRIMARY_H
