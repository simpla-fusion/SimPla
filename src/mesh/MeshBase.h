//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "../base/Object.h"
#include "../gtl/Log.h"

#include "MeshCommon.h"
#include "MeshBase.h"
#include "MeshEntity.h"

namespace simpla { namespace data_model { struct DataSpace; }}
namespace simpla { namespace io { struct IOStream; }}

namespace simpla { namespace mesh
{

class MeshBase : public base::Object, public std::enable_shared_from_this<MeshBase>
{
    int m_level_;
    unsigned long m_status_flag_ = 0;
public:

    SP_OBJECT_HEAD(MeshBase, base::Object);


    MeshBase() : m_level_(0), m_status_flag_(0) { }

    virtual    ~MeshBase() { }

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


    unsigned long const &status() const { return m_status_flag_; }

    void status(unsigned long l) { m_status_flag_ = l; }

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
 *   *      |       |  *(entity_id_range)*   |   |     *
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

    virtual box_type box(MeshEntityStatus entityStatus = SP_ES_VALID) const = 0;

    virtual MeshEntityRange select(box_type const &b,
                                   MeshEntityType entityType = VERTEX,
                                   MeshEntityStatus entityStatus = SP_ES_VALID) const = 0;

    virtual MeshEntityRange range(MeshEntityType entityType = VERTEX,
                                  MeshEntityStatus entityStatus = SP_ES_VALID) const = 0;

    virtual MeshEntityRange range(box_type const &b, MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t max_hash(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t hash(MeshEntityId const &) const = 0;

    virtual point_type point(MeshEntityId const &) const = 0;

    virtual point_type point_local_to_global(MeshEntityId s, point_type const &r) const = 0;

    virtual std::tuple<MeshEntityId, point_type>
            point_global_to_local(point_type const &g, int nId = 0) const = 0;

    virtual int get_adjacent_entities(MeshEntityType const &t, MeshEntityId const &,
                                      MeshEntityId *p = nullptr) const = 0;


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

    virtual std::tuple<data_model::DataSpace, data_model::DataSpace> data_space(MeshEntityType const &t) const = 0;

    virtual std::shared_ptr<MeshBase> clone() const = 0;

    virtual std::shared_ptr<MeshBase> extend(int const *od, size_type w = 2) const
    {
        auto res = this->clone();
        UNIMPLEMENTED;
        return res;
    };

    virtual std::shared_ptr<MeshBase> refine(box_type const &, int refine_ratio = 2) const
    {
        auto res = this->clone();
        UNIMPLEMENTED;
        return res;
    };

    virtual std::shared_ptr<MeshBase> coarsen(box_type const &, int refine_ratio = 2) const
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
        VERBOSE << " Mesh [ " << name() << " ] next time step, time = " << m_time_ / dt << " ." << std::endl;
        m_time_ += dt;
    };

private:
    Real m_time_ = 0.0;

};

}}//namespace simpla{namespace get_mesh{
#endif //SIMPLA_MESHBASE_H
