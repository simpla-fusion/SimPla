/**
 * @file MeshBlock.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_BLOCK_H
#define SIMPLA_BLOCK_H

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/utilities/log.h"
#include "MeshIds.h"

namespace simpla { namespace mesh
{

struct MeshBlock : public MeshIDs, public base::Object
{


    SP_OBJECT_HEAD(MeshBlock, base::Object)


    static constexpr int ndims = 3;

    enum
    {
        DEFAULT_GHOST_WIDTH = 2
    };
private:

    typedef MeshBlock this_type;
    typedef MeshIDs m;


public:
    using typename m::id_type;
    using typename m::id_tuple;
    using typename m::index_type;
    using typename m::range_type;
    using typename m::iterator;
    using typename m::index_tuple;
    using typename m::difference_type;


    typedef std::tuple<index_tuple, index_tuple> index_box_type;


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
 *	0 = m_idx_min_
 *	5 = m_idx_max_
 *
 *	1 = m_idx_memory_min_
 *	4 = m_idx_memory_max_
 *
 *	2 = m_idx_local_min_
 *	3 = m_idx_local_max_
 *
 *
 */

    index_tuple m_idx_min_;
    index_tuple m_idx_max_;
    index_tuple m_idx_local_min_;
    index_tuple m_idx_local_max_;
    index_tuple m_idx_memory_min_;
    index_tuple m_idx_memory_max_;


public:

    MeshBlock();

    MeshBlock(this_type const &other) = delete;

    virtual  ~MeshBlock();

public:


    int number_of_dims() const
    {
        int count = 0;
        for (int i = 0; i < ndims; ++i)
        {
            if (m_idx_max_[i] - m_idx_min_[i] > 1)++count;
        }
        return count;
    }

    size_t id_mask() const
    {
        id_type M0 = ((1UL << ID_DIGITS) - 1);
        id_type M1 = ((1UL << (MESH_RESOLUTION)) - 1);
        return FULL_OVERFLOW_FLAG
               | ((m_idx_max_[0] - m_idx_min_[0] > 1) ? M0 : M1)
               | (((m_idx_max_[1] - m_idx_min_[1] > 1) ? M0 : M1) << ID_DIGITS)
               | (((m_idx_max_[2] - m_idx_min_[2] > 1) ? M0 : M1) << (ID_DIGITS * 2));
    }


    void dimensions(index_tuple const &d)
    {
        m_idx_max_ = m_idx_min_ + d;
    }

    index_tuple dimensions() const
    {
        index_tuple res;

        res = m_idx_max_ - m_idx_min_;

        return std::move(res);
    }


    void index_box(index_tuple const &min, index_tuple const &max)
    {
        m_idx_min_ = min;
        m_idx_max_ = max;
    };


    index_box_type index_box() const
    {
        return (std::make_tuple(m_idx_min_, m_idx_max_));
    }


    index_box_type index_box(id_type const &s) const
    {
        return std::make_tuple(m::unpack_index(s - _DA), m::unpack_index(s + _DA));
    }

    index_box_type local_index_box() const
    {
        return (std::make_tuple(m_idx_local_min_, m_idx_local_max_));
    }

    index_box_type memory_index_box() const
    {
        return (std::make_tuple(m_idx_memory_min_, m_idx_memory_max_));
    }

    bool in_box(index_tuple const &x) const
    {
        return (m_idx_local_min_[1] <= x[1]) && (m_idx_local_min_[2] <= x[2]) && (m_idx_local_min_[0] <= x[0])  //
               && (m_idx_local_max_[1] > x[1]) && (m_idx_local_max_[2] > x[2]) && (m_idx_local_max_[0] > x[0]);

    }

    bool in_box(id_type s) const { return in_box(m::unpack_index(s)); }

    template<int IFORM>
    range_type range() const { return m::template make_range<IFORM>(m_idx_local_min_, m_idx_local_max_); }


    template<int IFORM>
    size_t max_hash() const
    {
        return static_cast<size_t>(m::hash(
                m::pack_index(m_idx_memory_max_ - 1, m::template sub_index_to_id<IFORM>(3UL)),
                m_idx_memory_min_, m_idx_memory_max_));
    }


    size_t hash(id_type const &s) const
    {
        return static_cast<size_t>(m::hash(s, m_idx_memory_min_, m_idx_memory_max_));
    }


    void decompose(index_tuple const &dist_dimensions, index_tuple const &dist_coord,
                   index_type gw = DEFAULT_GHOST_WIDTH);

    virtual void deploy();

    void update_boundary_box();

    index_box_type const &center_box() const { return m_center_box_; }

    std::vector<index_box_type> const &boundary_box() const { return m_boundary_box_; }

    std::vector<index_box_type> const &ghost_box() const { return m_ghost_box_; }

private:
    index_box_type m_center_box_;
    std::vector<index_box_type> m_boundary_box_;
    std::vector<index_box_type> m_ghost_box_;

}; // struct MeshBlock

}}//namespace simpla//namespace mesh
#endif //SIMPLA_BLOCK_H
