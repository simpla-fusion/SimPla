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

    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;


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

    typedef nTuple<Real, ndims> point_type;
    typedef nTuple<Real, ndims> vector_type;
    typedef nTuple<Real, 2, ndims> box_type;

    typedef nTuple<index_type, 2, ndims> index_box_type;


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

    point_type m_x_min_;
    point_type m_x_max_;
    point_type m_x_scale_;

    index_tuple m_idx_min_;
    index_tuple m_idx_max_;
    index_tuple m_idx_local_min_;
    index_tuple m_idx_local_max_;
    index_tuple m_idx_memory_min_;
    index_tuple m_idx_memory_max_;


    bool m_is_valid_ = false;
public:

    MeshBlock();

    MeshBlock(this_type const &other) = delete;

    virtual  ~MeshBlock();

//            base_type(other),
//            m_idx_min_(other.m_idx_min_),
//            m_idx_max_(other.m_idx_max_),
//            m_idx_local_min_(other.m_idx_local_min_),
//            m_idx_local_max_(other.m_idx_local_max_),
//            m_idx_memory_min_(other.m_idx_memory_min_),
//            m_idx_memory_max_(other.m_idx_memory_max_)
//    {
//
//    }


//    virtual void swap(this_type &other)
//    {
//        base_type::swap(other);
//
//        std::swap(m_idx_min_, other.m_idx_min_);
//        std::swap(m_idx_max_, other.m_idx_max_);
//        std::swap(m_idx_local_min_, other.m_idx_local_min_);
//        std::swap(m_idx_local_max_, other.m_idx_local_max_);
//        std::swap(m_idx_memory_min_, other.m_idx_memory_min_);
//        std::swap(m_idx_memory_max_, other.m_idx_memory_max_);
//        deploy();
//        other.deploy();
//
//    }


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

    template<typename TD>
    void dimensions(TD const &d)
    {
        index_tuple dims;
        dims = d;
        properties()["Geometry"]["Topology"]["Dimensions"] = dims;
    }

    index_tuple dimensions() const;

    virtual void box(box_type const &b);

    virtual box_type box() const;

    virtual box_type box(id_type const &s) const;

    virtual box_type local_box() const;

    template<typename T0, typename T1>
    void index_box(T0 const &min, T1 const &max)
    {
        m_idx_min_ = min;
        m_idx_max_ = max;
    };


    auto index_box() const
    DECL_RET_TYPE((traits::make_nTuple(m_idx_min_, m_idx_max_)))

    auto local_index_box() const
    DECL_RET_TYPE((traits::make_nTuple(m_idx_local_min_, m_idx_local_max_)))

    auto memory_index_box() const
    DECL_RET_TYPE((traits::make_nTuple(m_idx_memory_min_, m_idx_memory_max_)))

    bool in_box(index_tuple const &x) const
    {
        return (m_idx_local_min_[1] <= x[1]) && (m_idx_local_min_[2] <= x[2]) && (m_idx_local_min_[0] <= x[0])  //
               && (m_idx_local_max_[1] > x[1]) && (m_idx_local_max_[2] > x[2]) && (m_idx_local_max_[0] > x[0]);

    }

    bool in_box(id_type s) const { return in_box(m::unpack_index(s)); }

    template<int I>
    range_type range() const { return m::template make_range<I>(m_idx_local_min_, m_idx_local_max_); }


    template<int IFORM>
    auto max_hash() const
    DECL_RET_TYPE((m::hash(m::pack_index(m_idx_memory_max_ - 1, m::template sub_index_to_id<IFORM>(3UL)),
                           m_idx_memory_min_, m_idx_memory_max_)))


    size_t hash(id_type const &s) const
    {
        return static_cast<size_t>(m::hash(s, m_idx_memory_min_, m_idx_memory_max_));
    }


    void decompose(index_tuple const &dist_dimensions, index_tuple const &dist_coord,
                   index_type gw = DEFAULT_GHOST_WIDTH);

    virtual void deploy();

    void update_boundary_box();

    box_type const &center_box() const { return m_center_box_; }

    std::vector<box_type> const &boundary_box() const { return m_boundary_box_; }

    std::vector<box_type> const &ghost_box() const { return m_ghost_box_; }

private:
    box_type m_center_box_;
    std::vector<box_type> m_boundary_box_;
    std::vector<box_type> m_ghost_box_;


//    template<typename T0, typename T1>
//    index_type dist_to_box_(id_type const &s, T0 const &imin, T1 const &imax) const
//    {
//        auto idx = this->unpack_index(s);
//
//        index_type res = std::numeric_limits<index_type>::max();
//
//        for (int i = 0; i < 3; ++i)
//        {
//            if (imax[i] - imin[i] <= 1)
//            {
//
//                continue;
//            }
//            else
//            {
//                res = std::min(res, std::min(idx[i] - imin[i], imax[i] - idx[i]));
//
//            }
//        }
//
//        return res;
//    }
//
//public:
//
//    index_type idx_to_local_boundary(id_type const &s) const
//    {
//        return dist_to_box_(s, m_idx_local_min_, m_idx_local_max_);
//    }
//
//    index_type idx_to_boundary(id_type const &s) const
//    {
//        return dist_to_box_(s, m_idx_min_, m_idx_max_);
//
//    }


}; // struct MeshBlock
}//namespace mesh
}//namespace simpla
#endif //SIMPLA_BLOCK_H
