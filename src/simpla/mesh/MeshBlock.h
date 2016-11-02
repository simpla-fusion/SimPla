//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <simpla/SIMPLA_config.h>
#include <iomanip>
#include <simpla/data/DataBase.h>
#include <simpla/data/DataSpace.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/Object.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>
#include <simpla/toolbox/BoxUtility.h>
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

class MeshBlock :
        public toolbox::Object,
        public toolbox::Serializable,
        public toolbox::Printable,
        public std::enable_shared_from_this<MeshBlock>
{

public:

    SP_OBJECT_HEAD(MeshBlock, toolbox::Object)

    MeshBlock();

    MeshBlock(index_tuple const &lo, index_tuple const &hi, index_tuple const &gw, id_type space_id = 0);

    MeshBlock(int ndims, index_type const *lo, index_type const *hi, index_type const *gw, id_type space_id = 0);

    virtual ~MeshBlock();

    virtual void swap(MeshBlock &other);

    MeshBlock(MeshBlock const &other) = delete;

    MeshBlock(MeshBlock &&other) = delete;

    MeshBlock &operator=(MeshBlock const &other)= delete;

    /** for Printable @{*/
    virtual std::string const &name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    /** @}*/

    /** for Serializable @{*/

    virtual void load(const data::DataBase &) {};

    virtual void save(data::DataBase *) {};

    /** @}*/

    /**
      * @return   copy of this mesh but has different '''id'''
      */

    virtual std::shared_ptr<MeshBlock> clone() const =0;

    /**
     *
     *   inc_level <  0 => create a new mesh in coarser  index space
     *   inc_level == 0 => create a new mesh in the same index space
     *   inc_level >  0 => create a new mesh in finer    index space
     *
     *   dx_new= dx_old* 2^ (-inc_level)
     *   offset_new = b.first * 2^ (-inc_level)
     *   count_new  = b.second * 2^ (-inc_level) - offset_new
     */
    virtual std::shared_ptr<MeshBlock> create(index_box_type const &b, int inc_level = 1) const;


    /**
     * create a sub-mesh of this mesh, with same m_root_id
     * @param other_box
     */
    std::shared_ptr<MeshBlock> intersection(index_box_type const &other_box);

    size_type root_id() const;

    size_type space_id() const;

    int level() const;

    virtual bool is_overlap(index_box_type const &, int inc_level = 0);

    virtual bool is_overlap(box_type const &, int inc_level = 0);

    virtual bool is_overlap(MeshBlock const &, int inc_level = 0);

    /**
     *  Set unique ID of index space
     * @param id
     */


    void shift(index_type x, index_type y = 0, index_type z = 0);

    void shift(index_type const *);

    virtual void deploy();

    virtual bool is_deployed() const;

    virtual bool is_valid();

    size_tuple dimensions() const;

    size_tuple const &ghost_width() const;

    index_box_type const &global_index_box() const;

    index_box_type const &memory_index_box() const;

    index_box_type const &inner_index_box() const;

    index_box_type const &outer_index_box() const;

    box_type get_box(index_box_type const &b) const
    {
        return std::make_tuple(point(std::get<0>(b)), point(std::get<1>(b)));
    }

    virtual Real const *dx() const;

    virtual box_type box() const { return get_box(inner_index_box()); };

    virtual box_type global_box() const { return get_box(global_index_box()); };

    virtual box_type memory_box() const { return get_box(memory_index_box()); };

    virtual box_type inner_box() const { return get_box(inner_index_box()); };

    virtual box_type outer_box() const { return get_box(outer_index_box()); };

    virtual point_type point(MeshEntityId const &s) const { return point(unpack(s)); }

    virtual point_type point(index_tuple const &b) const;

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

    inline size_type hash(MeshEntityId const &id) const
    {
        return hash(id.x >> 1, id.y >> 1, id.z >> 1, m::node_id(id));
    }


    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s, MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }

    virtual index_tuple point_to_index(point_type const &g, int nId = 0) const
    {
        return m::unpack_index(std::get<0>(m::point_global_to_local(g, nId)));
    };

    virtual EntityIdRange range(MeshEntityType entityType = VERTEX, MeshZoneTag status = SP_ES_OWNED) const;

    virtual EntityIdRange range(MeshEntityType entityType, index_box_type const &b) const;

    virtual EntityIdRange range(MeshEntityType entityType, box_type const &b) const;

private:


    bool m_is_deployed_ = false;
    id_type m_space_id_ = 0;
    int m_level_ = 0;
    Real m_time_ = 0.0;
    int m_ndims_;


    size_tuple m_ghost_width_{{0, 0, 0}};        //!<     ghost width

    index_box_type m_g_box_{{0, 0, 0},
                            {1, 1, 1}};         //!<     global index block
    index_box_type m_m_box_{{0, 0, 0},
                            {1, 1, 1}};         //!<     memory index block
    index_box_type m_inner_box_{{0, 0, 0},
                                {1, 1, 1}};     //!<    inner block
    index_box_type m_outer_box_{{0, 0, 0},
                                {1, 1, 1}};     //!<    outer block


};

}} //namespace simpla{namespace mesh_as
#endif //SIMPLA_BOX_H
