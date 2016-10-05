//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "../base/Object.h"
#include "../toolbox/Log.h"

#include "MeshCommon.h"
#include "Chart.h"
#include "EntityRange.h"
#include "EntityId.h"

namespace simpla { namespace data_model { struct DataSpace; }}
namespace simpla { namespace io { struct IOStream; }}

namespace simpla { namespace mesh
{
enum { SP_MESH_AX_NORMAL = 0, SP_MESH_AX_CYCLE = 0x1, SP_MESH_AX_NULL = 0x3 };

class Chart : public base::Object
{
    int m_level_ = 0;
    int m_status_flag_ = 0;
    bool m_is_virtual_ = false;
public:

    SP_OBJECT_HEAD(Chart, base::Object);

    Chart(int l = 0) : m_level_(l), m_status_flag_(0), m_time_(0) {}

    Chart(Chart const &other) : m_level_(other.m_level_), m_status_flag_(other.m_status_flag_),
                                m_time_(other.m_time_) {}

    virtual ~Chart() {}

    virtual io::IOStream &save(io::IOStream &os) const
    {
        UNIMPLEMENTED;
        return os;
    };

    virtual io::IOStream &load(io::IOStream &is)
    {
        UNIMPLEMENTED;
        return is;
    };

    bool is_virtual() const { return m_is_virtual_; }

    /** status:   0000000zzyyxx
     *  xx : 00 NORMAL
     *       01 CYCLE
     *       11 NULL
     */
    int const &status() const { return m_status_flag_; }

    void status(int l) { m_status_flag_ = l; }

    int level() const { return m_level_; }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

/**
 *
 *   ********************outer box************
 *   *      |--------------------------|     *
 *   *      |       +-----box------+   |     *
 *   *      |       |              |   |     *
 *   *      |       |  **inner**   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *       *   |   |     *
 *   *      |       |  *(entity_id_range)    *
 *   *      |       |  2********   |   |     *
     *      |       |   boundary   |   |     *
 *   *    /---      +--------------+   |     *
     *   /  |             affected     |     *
     *ghost |--------------------------|     *
     *   \---                                *
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

    virtual index_tuple const &ghost_width() const = 0;

    virtual void ghost_width(index_tuple const &) = 0;

    virtual box_type box(MeshEntityStatus entityStatus = SP_ES_ALL) const = 0;

    virtual index_box_type index_box(box_type const &b) const = 0;

    virtual index_tuple offset() const = 0;

    virtual point_type origin_point() const = 0;

    virtual mesh::Chart &shift(index_tuple const &) { return *this; }

    virtual mesh::Chart &stretch(index_tuple const &) { return *this; }

    virtual EntityRange select(box_type const &b,
                               MeshEntityType entityType = VERTEX,
                               MeshEntityStatus entityStatus = SP_ES_ALL) const = 0;

    virtual EntityRange range(MeshEntityType entityType = VERTEX,
                              MeshEntityStatus entityStatus = SP_ES_ALL) const = 0;

    virtual EntityRange range(box_type const &b, MeshEntityType entityType = VERTEX) const = 0;

    virtual EntityRange range(index_box_type const &b, MeshEntityType entityType = VERTEX) const = 0;

    virtual size_type max_hash(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_type hash(MeshEntityId const &) const = 0;

    virtual point_type point(MeshEntityId const &) const = 0;

    virtual point_type point_local_to_global(MeshEntityId s, point_type const &r) const = 0;

    virtual std::tuple<MeshEntityId, point_type>
    point_global_to_local(point_type const &g, int nId = 0) const = 0;

    virtual int get_adjacent_entities(MeshEntityType t, MeshEntityId, MeshEntityId *p = nullptr) const = 0;

    int get_vertices(MeshEntityId const &s, point_type *p) const
    {
        int num = get_adjacent_entities(VERTEX, s);

        if (p != nullptr)
        {
            MeshEntityId neighbour[num];

            get_adjacent_entities(VERTEX, s, neighbour);

            for (int i = 0; i < num; ++i) { p[i] = point(neighbour[i]); }
        }
        return num;

    }

    virtual std::tuple<data_model::DataSpace, data_model::DataSpace>
    data_space(MeshEntityType const &t, MeshEntityStatus status = SP_ES_OWNED) const = 0;

    virtual std::shared_ptr<Chart> clone(std::string const &name = "") const = 0;

    template<typename TM>
    std::shared_ptr<TM> clone_as(std::string const &name = "") const
    {
        assert(is_a<TM>());
        return std::dynamic_pointer_cast<TM>(clone(name));
    };

    virtual std::shared_ptr<Chart> refine(box_type const &, int refine_ratio = 2) const
    {
        auto res = this->clone();
        UNIMPLEMENTED;
        return res;
    };

    virtual std::shared_ptr<Chart> coarsen(box_type const &, int refine_ratio = 2) const
    {
        auto res = this->clone();
        UNIMPLEMENTED;
        return res;
    };


    //------------------------------------------------------------------------------------------------------------------
    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; };

    void next_step(Real dt)
    {
        VERBOSE << " Mesh  next time step, time = " << m_time_ << " dt = " << dt << std::endl;
        m_time_ += dt;
    };


private:
    Real m_time_ = 0.0;

};

}}//namespace simpla{namespace get_mesh{
#endif //SIMPLA_MESHBASE_H

