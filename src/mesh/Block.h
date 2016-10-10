//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <iomanip>
#include "MeshCommon.h"
#include "../toolbox/Object.h"
#include "EntityId.h"
#include "../toolbox/DataSpace.h"

namespace simpla { namespace mesh
{
/**
 *  block represent a n-dims block in the index space;
 *
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
`*
 *   ********************outer box************
 *   *      |--------------------------|     *
 *   *      |       +-----box------+   |     *
 *   *      |       |              |   |     *
 *   *      |       |  **inner**   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *(entity_id_range)    *
 *   *      |       |  2********   |   |     *
 *   *    /---      +--------------+   |     *
 *   *      |       |   boundary   |   |     *
 *   *   /  |             affected     |     *
 *   *ghost |--------------------------|     *
 *   *   \---                                *
 *   *****************************************
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

class Block : public toolbox::Object
{
public:
    SP_OBJECT_HEAD(Block, toolbox::Object)
    using toolbox::Object::id_type;

    static constexpr int ndims = 3;

    Block();

    Block(Block const &other);

    Block(Block &&other);

    virtual ~Block();

    Block &operator=(Block const &other)
    {
        Block(other).swap(*this);
        return *this;
    }

    void swap(Block &other);

    virtual std::tuple<toolbox::DataSpace, toolbox::DataSpace>
    data_space(MeshEntityType const &t, MeshEntityStatus status = SP_ES_OWNED) const;

    void processer_id(int id) { processer_id_ = id; }

    int processer_id() const { return processer_id_; }

    void global_id(size_type id) { m_global_id_ = id; }

    size_type global_id() const { return m_global_id_; }

    int level() const { return m_level_; }

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

    void intersection(Block const &other);

    virtual void refine(int ratio = 1);

    virtual void coarsen(int ratio = 1);

    virtual void deploy();

    bool is_deployed() const { return m_is_deployed_; }

    index_tuple const &dimensions() const { return m_b_dimensions_; }

    index_tuple const &local_dimensions() const { return m_l_dimensions_; }

    index_tuple const &local_offset() const { return m_l_offset_; }

    index_tuple const &global_dimensions() const { return m_g_dimensions_; }

    index_tuple const &blobal_offset() const { return m_g_offset_; }

    index_box_type local_index_box() const
    {
        index_tuple lower = m_l_offset_;
        index_tuple upper;
        upper = lower + m_b_dimensions_;
        return std::make_tuple(lower, upper);
    }

    index_box_type global_index_box() const
    {
        index_tuple lower = m_g_offset_;
        index_tuple upper;
        upper = lower + m_b_dimensions_;
        return std::make_tuple(lower, upper);
    }


    bool empty() const { return size() == 0; }

    size_type size() const { return m_l_dimensions_[0] * m_l_dimensions_[1] * m_l_dimensions_[2]; }

    inline size_type hash(size_type i, size_type j = 0, size_type k = 0) const
    {
        return (i * m_l_dimensions_[1] + j) * m_l_dimensions_[2] + k;
    }

    index_tuple unhash(size_type s) const
    {
        index_tuple res;
        res[2] = s % m_l_dimensions_[2];
        res[1] = (s / m_l_dimensions_[2]) % m_l_dimensions_[1];
        res[0] = s / (m_l_dimensions_[2] * m_l_dimensions_[1]);

        return std::move(res);
    }

    typedef MeshEntityIdCoder m;
    typedef MeshEntityId id;

    id pack(size_type i, size_type j = 0, size_type k = 0, int nid = 0) const { return m::pack_index(i, j, k, nid); }

    index_tuple unpack(id const &s) const { return m::unpack_index(s); }

    void for_each(std::function<void(size_type, size_type, size_type)> const &fun) const;

    void for_each(std::function<void(size_type)> const &fun) const;

    void for_each(std::function<void(id const &)> const &fun, int iform = VERTEX) const;

private:
    int processer_id_ = 0;
    size_type m_global_id_ = 0;
    int m_level_ = 0;
    bool m_is_deployed_ = false;

    index_tuple m_b_dimensions_{{1, 1, 1}};      //!<   dimensions of box
    index_tuple m_ghost_width_{{0, 0, 0}};          //!<     start index in the local  space
    index_tuple m_l_dimensions_{{1, 1, 1}};      //!<   dimensions of local index space
    index_tuple m_l_offset_{{0, 0, 0}};          //!<     start index in the local  space
    index_tuple m_g_dimensions_{{1, 1, 1}};     //!<   dimensions of global index space
    index_tuple m_g_offset_{{0, 0, 0}};         //!<   start index of global index space

};


}} //namespace simpla{namespace mesh
#endif //SIMPLA_BOX_H
