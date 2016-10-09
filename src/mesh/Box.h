//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <iomanip>
#include "MeshCommon.h"
#include "../toolbox/Object.h"
#include "EntityId.h"

namespace simpla { namespace mesh
{

/**
*
*   -----------------------------5
*   |                            |
*   |     ---------------4       |
*   |     |              |       |
*   |     |  ********3   |       |
*   |     |  *       *   |       |
*   |     |  *       *   |       |
*   |     |  *       *   |       |
*   |     |  2********   |       |
*   |     1---------------       |
*   0-----------------------------
*
*	5-0 = dimensions
*	4-1 = e-d = ghosts
*	2-1 = counts
*
*	0 = id_begin
*	5 = id_end
*
*	1 = id_local_outer_begin
*	4 = id_local_outer_end
*
*	2 = id_local_inner_begin
*	3 = id_local_inner_end
*
*
*/
/**
 *  Box represent a n-dims box in the index space;
 *
 */
class Box : public toolbox::Object
{
public:
    SP_OBJECT_HEAD(Box, toolbox::Object)

    static constexpr int ndims = 3;

    int processer_id_ = 0;
    size_type m_global_id_;
    int m_level_ = 0;

    index_tuple m_b_dimensions_ = {1, 1, 1};      //!<   dimensions of box
    index_tuple m_ghost_width_ = {0, 0, 0};          //!<     start index in the local  space
    index_tuple m_l_dimensions_ = {1, 1, 1};      //!<   dimensions of local index space
    index_tuple m_l_offset_ = {0, 0, 0};          //!<     start index in the local  space
    index_tuple m_g_dimensions_ = {1, 1, 1};      //!<   dimensions of global index space
    index_tuple m_g_offset_ = {0, 0, 0};          //!<   start index of global index space

    bool m_is_deployed_ = false;

    Box() {}

    Box(Box const &other) :
            processer_id_(other.processer_id_),
            m_b_dimensions_(other.m_b_dimensions_),
            m_l_dimensions_(other.m_l_dimensions_),
            m_l_offset_(other.m_l_offset_),
            m_g_dimensions_(other.m_g_dimensions_),
            m_g_offset_(other.m_g_offset_),
            m_is_deployed_(false) {};

    Box(Box &&other) :
            processer_id_(other.processer_id_),
            m_b_dimensions_(other.m_b_dimensions_),
            m_l_dimensions_(other.m_l_dimensions_),
            m_l_offset_(other.m_l_offset_),
            m_g_dimensions_(other.m_g_dimensions_),
            m_g_offset_(other.m_g_offset_),
            m_is_deployed_(other.m_is_deployed_) {};

    ~Box() {}

    Box &operator=(Box const &other)
    {
        Box(other).swap(*this);
        return *this;
    }

    void swap(Box const &other)
    {
        std::swap(processer_id_, other.processer_id_);
        std::swap(m_b_dimensions_, other.m_b_dimensions_);
        std::swap(m_l_dimensions_, other.m_l_dimensions_);
        std::swap(m_l_offset_, other.m_l_offset_);
        std::swap(m_g_dimensions_, other.m_g_dimensions_);
        std::swap(m_g_offset_, other.m_g_offset_);
        std::swap(m_is_deployed_, other.m_is_deployed_);
    }

    void dimensions(index_tuple const &d)
    {
        assert(!m_is_deployed_);
        m_b_dimensions_ = d;
    }

    void ghost_width(index_tuple const &d)
    {
        assert(!m_is_deployed_);
        m_ghost_width_ = d;
    }

    void shift(nTuple<int, ndims> const &offset)
    {
        assert(!m_is_deployed_);
        m_g_offset_ += offset;
    };

    void stretch(index_tuple const &a)
    {
        assert(!m_is_deployed_);
        m_b_dimensions_ *= a;
    };

    void intersection(Box const &other)
    {
        assert(!m_is_deployed_);

        assert(m_global_id_ == other.m_global_id_);


        for (int i = 0; i < ndims; ++i)
        {
            size_type l_lower = m_g_offset_[i];
            size_type l_upper = m_g_offset_[i] + m_b_dimensions_[i];
            size_type r_lower = other.m_g_offset_[i];
            size_type r_upper = other.m_g_offset_[i] + other.m_b_dimensions_[i];
            l_lower = std::max(l_lower, r_lower);
            l_upper = std::min(l_upper, l_upper);
            m_b_dimensions_[i] = (l_upper > l_lower) ? (l_upper - l_lower) : 0;
            m_g_offset_[i] = l_lower;
        }
    };

    void refine(int ratio = 1)
    {
        assert(!m_is_deployed_);
        ++m_level_;
        for (int i = 0; i < ndims; ++i)
        {
            m_b_dimensions_[i] <<= ratio;
            m_g_dimensions_[i] <<= ratio;
            m_g_offset_[i] <<= ratio;
            m_l_dimensions_[i] = 0;
            m_l_offset_[i] = 0;
        }
    }

    void coarsen(int ratio = 1)
    {
        assert(!m_is_deployed_);
        --m_level_;
        for (int i = 0; i < ndims; ++i)
        {
            int mask = (1 << ratio) - 1;
            assert(m_b_dimensions_[i] & mask == 0);
            assert(m_g_dimensions_[i] & mask == 0);
            assert(m_g_offset_[i] & mask == 0);

            m_b_dimensions_[i] >>= ratio;
            m_g_dimensions_[i] >>= ratio;
            m_g_offset_[i] >>= ratio;
            m_l_dimensions_[i] = 0;
            m_l_offset_[i] = 0;
        }

    }

    void deploy()
    {

        for (int i = 0; i < 3; ++i)
        {
            if (m_l_offset_[i] < m_ghost_width_[i] ||
                m_l_dimensions_[i] < m_ghost_width_[i] * 2 + m_b_dimensions_[i])
            {
                m_l_offset_[i] = m_ghost_width_[i];
                m_l_dimensions_[i] = m_ghost_width_[i] * 2 + m_b_dimensions_[i];
            }

            if (m_b_dimensions_[i] < 1 ||
                m_l_dimensions_[i] < m_b_dimensions_[i] + m_l_offset_[i] ||
                m_g_dimensions_[i] < m_b_dimensions_[i] + m_g_offset_[i])
            {
                m_is_deployed_ = false;
            }
        }
        m_is_deployed_ = true;
    }

    bool is_deployed() const { return m_is_deployed_; }

    size_type global_id() const { return m_global_id_; }

    int level() const { return m_level_; }

    index_tuple const &dimensions() const { return m_b_dimensions_; }

    index_tuple const &local_dims() const { return m_l_dimensions_; }

    index_tuple const &local_offset() const { return m_l_offset_; }

    index_tuple const &global_dims() const { return m_g_dimensions_; }

    index_tuple const &blobal_offset() const { return m_g_offset_; }

    bool empty() const { return size() == 0; }

    size_type size() const
    {
        return m_l_dimensions_[0] * m_l_dimensions_[1] * m_l_dimensions_[2];
    }


    template<typename TFun>
    void for_each(TFun const &fun) const
    {
#pragma omp parallel for
        for (size_type i = m_l_offset_[0]; i < m_l_offset_[0] + m_b_dimensions_[0]; ++i)
            for (size_type j = m_l_offset_[1]; j < m_l_offset_[1] + m_b_dimensions_[1]; ++j)
                for (size_type k = m_l_offset_[2]; k < m_l_offset_[2] + m_b_dimensions_[2]; ++k)
                {
                    fun(i, j, k);
                }
    }
};
}} //namespace simpla{namespace mesh
#endif //SIMPLA_BOX_H
