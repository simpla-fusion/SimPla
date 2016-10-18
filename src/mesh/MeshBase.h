//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <iomanip>
#include "../toolbox/Object.h"
#include "../toolbox/DataSpace.h"
#include "MeshCommon.h"
#include "EntityId.h"

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

class MeshBase : public toolbox::Object
{
public:
    SP_OBJECT_HEAD(MeshBase, toolbox::Object)
    using toolbox::Object::id_type;

    static constexpr int ndims = 3;
    using toolbox::Object::id;

    MeshBase();

    MeshBase(MeshBase const &other);

    MeshBase(MeshBase &&other);

    virtual ~MeshBase();

    MeshBase &operator=(MeshBase const &other)
    {
        MeshBase(other).swap(*this);
        return *this;
    }

    virtual std::shared_ptr<MeshBase> clone() const { return std::make_shared<MeshBase>(*this); };

    void swap(MeshBase &other);

    virtual std::tuple<toolbox::DataSpace, toolbox::DataSpace>
    data_space(MeshEntityType const &t, MeshZoneTag status = SP_ES_OWNED) const;

    void processer_id(int id) { processer_id_ = id; }

    int processer_id() const { return processer_id_; }


    /**
     *  Set ID of space
     * @param id
     */
    void space_id(size_type id) { m_index_space_id_ = id; }

    size_type space_id() const { return m_index_space_id_; }

    int level() const { return m_level_; }

    void dimensions(index_tuple const &d)
    {
        assert(!m_is_deployed_);
        m_block_count_ = d;
    }

    void ghost_width(index_tuple const &d)
    {
        assert(!m_is_deployed_);
        m_ghost_width_ = d;
    }

    virtual void shift(index_tuple const &offset)
    {
        assert(!m_is_deployed_);
        m_g_start_ += offset;
    };

    virtual void scale(index_tuple const &a)
    {
        assert(!m_is_deployed_);
        for (int i = 0; i < ndims; ++i) { if (m_block_count_[i] > 1) { m_block_count_[i] *= a[i]; }}

    };

    virtual void stretch(index_tuple const &a)
    {
        assert(!m_is_deployed_);
        for (int i = 0; i < ndims; ++i)
        {
            if (m_block_count_[i] > 1) { m_block_count_[i] = static_cast<size_type>(a[i]); }
        }
    };

    virtual bool intersection(const box_type &other);

    virtual bool intersection(const index_box_type &other);

    virtual bool intersection_outer(const index_box_type &other);

    virtual void refine(int ratio = 1);

    virtual void coarsen(int ratio = 1);

    virtual void deploy();

    virtual box_type box() const
    {
        point_type lower, upper;
        lower = m_g_start_;
        upper = m_g_start_ + m_block_count_;
        return std::make_tuple(lower, upper);
    };

    virtual box_type outer_box() const
    {
        point_type lower, upper;
        lower = m_g_start_ - m_l_start_;
        upper = m_g_start_ + m_l_dimensions_;
        return std::make_tuple(lower, upper);
    }

    virtual point_type point(MeshEntityId const &s) const
    {
        point_type p;
        p = unpack(s);
        return std::move(p);
    }

    virtual std::tuple<MeshEntityId, point_type> point_global_to_local(point_type const &p, int iform = 0) const
    {
        return m::point_global_to_local(p, iform);
    }


    bool is_deployed() const { return m_is_deployed_; }

    size_tuple const &dimensions() const { return m_block_count_; }

    size_tuple const &local_dimensions() const { return m_l_dimensions_; }

    index_tuple const &local_offset() const { return m_l_start_; }

    size_tuple const &global_dimensions() const { return m_g_dimensions_; }

    index_tuple const &global_offset() const { return m_g_start_; }

    size_tuple const &ghost_width() const { return m_ghost_width_; }

    index_box_type local_index_box() const
    {
        index_tuple lower = m_l_start_;
        index_tuple upper;
        upper = lower + m_block_count_;
        return std::make_tuple(lower, upper);
    }

    index_box_type index_box() const
    {
        index_tuple lower = m_g_start_;
        index_tuple upper = lower + m_block_count_;
        return std::make_tuple(lower, upper);
    }

    index_box_type outer_index_box() const
    {
        index_tuple lower = m_g_start_ - m_l_start_;
        index_tuple upper = lower + m_l_dimensions_;
        return std::make_tuple(lower, upper);
    }


    bool empty() const { return size() == 0; }

    size_type size() const { return (m_l_dimensions_[0] * m_l_dimensions_[1] * m_l_dimensions_[2]); }

    size_type number_of_entities(int iform) const { return size() * ((iform == VERTEX || iform == VOLUME) ? 1 : 3); }

    inline size_type hash(index_type i, index_type j = 0, index_type k = 0) const
    {
        return static_cast<size_type>(((i + m_l_start_[0] - m_g_start_[0]) * m_l_dimensions_[1] +
                                       (j + m_l_start_[1] - m_g_start_[1])) * m_l_dimensions_[2] +
                                      k + m_l_start_[2] - m_g_start_[2]);
    }

    inline size_type hash(index_tuple const &id) const { return hash(id[0], id[1], id[2]); }

    index_tuple unhash(size_type s) const
    {
        index_tuple res;
        res[2] = s % m_l_dimensions_[2];
        res[1] = (s / m_l_dimensions_[2]) % m_l_dimensions_[1];
        res[0] = s / (m_l_dimensions_[2] * m_l_dimensions_[1]);

        return std::move(res);
    }


    typedef MeshEntityIdCoder m;

    inline size_type hash(MeshEntityId const &id) const
    {
        m::hash2(id, m_g_min_, m_l_dimensions_);
    }

    MeshEntityId pack(size_type i, size_type j = 0, size_type k = 0, int nid = 0) const
    {
        return m::pack_index(i, j, k, nid);
    }

    index_tuple unpack(MeshEntityId const &s) const { return m::unpack_index(s); }

//    void foreach(std::function<void(index_type, index_type, index_type)> const &fun) const;
//
//    void foreach(std::function<void(index_type)> const &fun) const;
//
//    void foreach(int iform, std::function<void(MeshEntityId const &)> const &) const;
//
//    template<typename TFun>
//    void foreach(int iform, TFun const &fun) const
//    {
//        int n = (iform == VERTEX || iform == VOLUME) ? 1 : 3;
//#pragma omp parallel for
//        for (index_type i = 0; i < m_block_count_[0]; ++i)
//            for (index_type j = 0; j < m_block_count_[1]; ++j)
//                for (index_type k = 0; k < m_block_count_[2]; ++k)
//                    for (int l = 0; l < n; ++l)
//                    {
//                        fun(pack(m_l_start_[0] + i, m_l_start_[1] + j, m_l_start_[2] + k, l));
//                    }
//    }


    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s,
                                      MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }

    virtual index_tuple
    point_to_index(point_type const &g, int nId = 0) const
    {
        return m::unpack_index(std::get<0>(m::point_global_to_local(g, nId)));
    };


    virtual EntityRange range(MeshEntityType entityType = VERTEX, MeshZoneTag status = SP_ES_OWNED) const;

    virtual EntityRange range(MeshEntityType entityType, index_box_type const &b) const;

    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; }

    void next_step(Real dt) { m_time_ += dt; }

private:
    int processer_id_ = 0;
    size_type m_index_space_id_ = 0;
    int m_level_ = 0;
    bool m_is_deployed_ = false;
    Real m_time_ = 0.0;

    size_tuple m_block_count_{{1, 1, 1}};        //!<   dimensions of local box

    index_tuple m_l_start_{{0, 0, 0}};          //!<     start index in the local  space

    size_tuple m_l_dimensions_{{1, 1, 1}};       //!<   dimensions of local index space

    index_tuple m_g_start_{{0, 0, 0}};        //!<   start index of global index space

    size_tuple m_g_dimensions_{{1, 1, 1}};       //!<   dimensions of global index space

    index_tuple m_g_min_{{0, 0, 0}};          //!<   local minum  global space index

    size_tuple m_ghost_width_{{0, 0, 0}};        //!<     ghost width

};

}} //namespace simpla{namespace mesh
#endif //SIMPLA_BOX_H
