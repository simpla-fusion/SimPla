//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <iomanip>
#include "../toolbox/Object.h"
#include "../toolbox/DataSpace.h"
#include "../toolbox/BoxUtility.h"
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
 *
 *   ********************global **************
 *   *      +---------memory  ---------+     *
 *   *      |       +-----outer ---+   |     *
 *   *      |       |              |   |     *
 *   *      |       |  **inner**   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *********   |   |     *
 *   *      |       +--------------+   |     *
 *   *      |                          |     *
 *   *      |                          |     *
 *   *      +--------------------------+     *
 *   *                                       *
 *   *****************************************
 *
 *    global > memory> outer > inner
 *    inner <  ghost < outer
 *
 */

class MeshBase : public toolbox::Object
{

private:
    bool m_is_deployed_ = false;
    int m_processer_id_ = 0;
    size_type m_space_id_ = 0;
    int m_level_ = 0;
    Real m_time_ = 0.0;


    size_tuple m_ghost_width_{{0, 0, 0}};        //!<     ghost width

    index_box_type m_g_box_{{{0, 0, 0}, {1, 1, 1}}};         //!<     global index block
    index_box_type m_m_box_{{{0, 0, 0}, {1, 1, 1}}};         //!<     memory index block
    index_box_type m_inner_box_{{{0, 0, 0}, {1, 1, 1}}};     //!<    inner block
    index_box_type m_outer_box_{{{0, 0, 0}, {1, 1, 1}}};     //!<    outer block
//    /**
//     *  stride for memory address hash
//     * index id[3]
//     *  pos = (id[0]-m_m_lower[0])*m_m_stride_[0] +
//     *        (id[1]-m_m_lower[1])*m_m_stride_[1] +
//     *        (id[2]-m_m_lower[2])*m_m_stride_[2]
//     */
//    size_tuple m_m_stride_{{1, 1, 1}};


public:

    SP_OBJECT_HEAD(MeshBase, toolbox::Object)

    using toolbox::Object::id_type;
    using toolbox::Object::id;

    static constexpr int ndims = 3;

    MeshBase();

    MeshBase(MeshBase const &other);

    MeshBase(MeshBase &&other);

    virtual ~MeshBase();

    MeshBase &operator=(MeshBase const &other)
    {
        MeshBase(other).swap(*this);
        return *this;
    }

    virtual std::ostream &print(std::ostream &os, int indent) const;


    /**
     * @return  a copy of this mesh
     */
    virtual std::shared_ptr<MeshBase> clone() const;


    /**
     * refine mesh with 2^n ratio
     *   n<0 => coarsen
     *   n=0 => clone
     *   n>0 => refine
     * @param n
     * @param flag RFU (reserved for future use)
     * @return
     */


    virtual void refine(index_box_type const &, int n, int flag = 0);


    /**
     * return block in this and other ,  if two block are not intersected return nullptr
     * @param flag
     * @return
     */
    virtual void intersection(index_box_type const &);


    void swap(MeshBase &other);

    virtual std::tuple<toolbox::DataSpace, toolbox::DataSpace>
    data_space(MeshEntityType const &t, MeshZoneTag status = SP_ES_OWNED) const;

    void processer_id(int id) { m_processer_id_ = id; }

    int processer_id() const { return m_processer_id_; }


    /**
     *  Set ID of space
     * @param id
     */
    void space_id(size_type id) { m_space_id_ = id; }

    size_type space_id() const { return m_space_id_; }

    int level() const { return m_level_; }

    void dimensions(index_tuple const &d) { if (!m_is_deployed_) { std::get<1>(m_g_box_) = std::get<0>(m_g_box_) + d; }}

    void ghost_width(index_tuple const &d) { if (!m_is_deployed_) { m_ghost_width_ = d; }}

    virtual void shift(index_tuple const &offset) { if (!m_is_deployed_) { std::get<0>(m_g_box_) += offset; }};

    virtual void stretch(index_tuple const &a) { if (!m_is_deployed_) { std::get<1>(m_g_box_) += a; }};

    virtual void deploy();


    bool is_deployed() const { return m_is_deployed_; }

    virtual bool is_valid()
    {
        return m_is_deployed_ &&
               toolbox::is_valid(m_g_box_) &&
               toolbox::is_valid(m_m_box_) &&
               toolbox::is_valid(m_inner_box_) &&
               toolbox::is_valid(m_outer_box_);
    }

    size_tuple const &ghost_width() const { return m_ghost_width_; }

    index_box_type const &global_index_box() const { return m_g_box_; }

    index_box_type const &memory_index_box() const { return m_m_box_; }

    index_box_type const &inner_index_box() const { return m_inner_box_; }

    index_box_type const &outer_index_box() const { return m_outer_box_; }

    box_type get_box(index_box_type const &b) const
    {
        return std::make_tuple(point(std::get<0>(b)), point(std::get<1>(b)));
    }


    virtual box_type box() const { return get_box(m_inner_box_); };

    virtual box_type global_box() const { return get_box(m_g_box_); };

    virtual box_type memory_box() const { return get_box(m_m_box_); };

    virtual box_type inner_box() const { return get_box(m_inner_box_); };

    virtual box_type outer_box() const { return get_box(m_outer_box_); };

    virtual point_type point(MeshEntityId const &s) const { return point(unpack(s)); }

    virtual point_type point(index_tuple const &b) const { return toolbox::convert(b); }


    virtual std::tuple<MeshEntityId, point_type> point_global_to_local(point_type const &p, int iform = 0) const
    {
        return m::point_global_to_local(p, iform);
    }


    bool empty() const { return max_hash() == 0; }


    size_type number_of_entities(int iform) const
    {
        return max_hash() * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
    }

    size_type max_hash() const { return toolbox::size(m_m_box_); }

    typedef MeshEntityIdCoder m;


    MeshEntityId pack(size_type i, size_type j = 0, size_type k = 0, int nid = 0) const
    {
        return m::pack_index(i, j, k, nid);
    }

    index_tuple unpack(MeshEntityId const &s) const { return m::unpack_index(s); }


    inline size_type hash(index_type i, index_type j = 0, index_type k = 0, int nid = 0) const
    {
        return static_cast<size_type>(m::hash(i, j, k, nid, std::get<0>(m_m_box_), std::get<1>(m_m_box_)));
    }


    inline size_type hash(index_tuple const &id) const { return hash(id[0], id[1], id[2]); }

    inline size_type hash(MeshEntityId const &id) const { return hash(unpack(id)); }

//    index_tuple unhash(size_type s, int nid = 0) const
//    {
//        index_tuple res;
//
//#ifndef SP_ARRAY_ORDER_FORTRAN
//        // C-Order Array
//        // m_m_stride_[0] >=  m_m_stride_[1]>=  m_m_stride_[2]
//        res[0] = s / m_m_stride_[0];
//        res[1] = (s % m_m_stride_[0]) / m_m_stride_[1];
//        res[2] = (s % m_m_stride_[1]) / m_m_stride_[2];
//#else
//
//#endif
//
//        return std::move(res);
//    }


    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s, MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }

    virtual index_tuple point_to_index(point_type const &g, int nId = 0) const
    {
        return m::unpack_index(std::get<0>(m::point_global_to_local(g, nId)));
    };

    virtual EntityRange range(MeshEntityType entityType = VERTEX, MeshZoneTag status = SP_ES_OWNED) const;

    virtual EntityRange range(MeshEntityType entityType, index_box_type const &b) const;

    virtual EntityRange range(MeshEntityType entityType, box_type const &b) const;

    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; }

    void next_step(Real dt) { m_time_ += dt; }


};

}} //namespace simpla{namespace mesh
#endif //SIMPLA_BOX_H
