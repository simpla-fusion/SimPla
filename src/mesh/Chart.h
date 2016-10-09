//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "../toolbox/Object.h"
#include "../toolbox/Log.h"

#include "MeshCommon.h"
#include "Block.h"
#include "EntityRange.h"
#include "EntityId.h"

namespace simpla { namespace mesh
{

class Chart : public Block
{
    point_type m_lower_{0, 0, 0};
    point_type m_upper_{1, 1, 1};
public:

    SP_OBJECT_HEAD(Chart, Block);

    Chart() {}

    Chart(Chart const &other) : Block(other) {}

    virtual ~Chart() {}

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual point_type point(MeshEntityId const &) const = 0;

    virtual point_type point(index_tuple const &) const
    {
        
    };

    virtual void deploy()
    {
        Block::deploy();
        auto id_box = global_index_box();
        box(point(std::get<0>(id_box)), point(std::get<1>(id_box)));
    }

    void box(point_type const &x0, point_type const &x1)
    {
        m_lower_ = x0;
        m_upper_ = x1;
    }

    void box(box_type const &b) { std::tie(m_lower_, m_upper_) = b; }

    box_type box(box_type const &b) const { return std::make_tuple(m_lower_, m_upper_); }

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

