//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "../base/Object.h"

#include "Mesh.h"
#include "MeshBase.h"
#include "MeshEntity.h"


namespace simpla { namespace mesh
{
class MeshAtlas;

class MeshBase : public base::Object
{
    int m_level_;
    unsigned long m_status_flag_ = 0;
public:

    SP_OBJECT_HEAD(MeshBase, base::Object);

    enum STATUS { LOCAL, ADJOINT, DISJOINT };

    MeshBase() : m_level_(0),m_status_flag_(LOCAL) { }

    virtual    ~MeshBase() { }

    int const &level() const { return m_level_; }

    void level(int l) { m_level_ = l; }

    unsigned long const &status() const { return m_status_flag_; }

    void status(unsigned long l) { m_status_flag_ = l; }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual box_type box() const = 0;

    virtual box_type outer_box() const { return box(); }

    virtual MeshEntityRange range(MeshEntityType entityType = VERTEX) const = 0;

    virtual MeshEntityRange ghost_range(MeshEntityType entityType = VERTEX) const
    {
        UNIMPLEMENTED;
        return MeshEntityRange();
    };

    virtual MeshEntityRange outer_range(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t max_hash(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t hash(MeshEntityId const &) const = 0;

    virtual point_type point(MeshEntityId const &) const = 0;

    virtual point_type point_local_to_global(MeshEntityId s, point_type const &r) const = 0;

    virtual std::tuple<MeshEntityId, point_type>
            point_global_to_local(point_type const &g, int nId = 0) const = 0;

    virtual int get_adjacent_entities(MeshEntityId const &, MeshEntityType t, MeshEntityId *p = nullptr) const = 0;

    virtual std::shared_ptr<MeshBase> refine(box_type const &b, int flag = 0) const = 0;


    int get_vertices(MeshEntityId const &s, point_type *p) const
    {
        int num = get_adjacent_entities(s, VERTEX);

        if (p != nullptr)
        {
            id_type neighbour[num];

            get_adjacent_entities(s, VERTEX, neighbour);

            for (int i = 0; i < num; ++i) { p[i] = point(neighbour[i]); }
        }
        return num;

    }

    virtual std::tuple<data_model::DataSpace, data_model::DataSpace> data_space(MeshEntityType const &t) const = 0;

};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHBASE_H
