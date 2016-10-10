//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_SPACEFILLINGCURVE_H
#define SIMPLA_SPACEFILLINGCURVE_H

#include "Block.h"

namespace simpla { namespace mesh
{

struct SpaceFillingCurveCOrder
{
    Block const &box;
    // C-order

    inline constexpr size_type hash(size_type i, size_type j = 0, size_type k = 0) const
    {
        return (i * box.m_l_dimensions_[1] + j) * box.m_l_dimensions_[2] + k;
    }

    index_tuple unhash(size_type s) const
    {
        index_tuple res;
        res[2] = s % box.m_l_dimensions_[2];
        res[1] = (s / box.m_l_dimensions_[2]) % box.m_l_dimensions_[1];
        res[0] = s / (box.m_l_dimensions_[2] * box.m_l_dimensions_[1]);

        return std::move(res);
    }

    inline constexpr size_type IX(size_type s) const
    {
        return s + box.m_l_dimensions_[2] *
                   box.m_l_dimensions_[1];
    }

    inline constexpr size_type DX(size_type s) const
    {
        return s - box.m_l_dimensions_[2] *
                   box.m_l_dimensions_[1];
    }

    inline constexpr size_type IY(size_type s) const { return s + box.m_l_dimensions_[1]; }

    inline constexpr size_type DY(size_type s) const { return s - box.m_l_dimensions_[1]; }

    inline constexpr size_type IZ(size_type s) const { return s + 1; }

    inline constexpr size_type DZ(size_type s) const { return s - 1; }

    template<typename TFun>
    void for_each(TFun const &fun) const
    {
#pragma omp parallel for
        for (int i = 0; i < box.m_b_dimensions_[0]; ++i)
            for (int j = 0; j < box.m_b_dimensions_[1]; ++j)
                for (int k = 0; k < box.m_b_dimensions_[2]; ++k)
                {
                    fun(hash(i + box.m_l_offset_[0], j + box.m_l_offset_[1], k + box.m_l_offset_[2]));
                }
    }


};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_SPACEFILLINGCURVE_H
