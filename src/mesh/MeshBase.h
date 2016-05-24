//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "Mesh.h"
#include "MeshEntity.h"
#include "../base/Object.h"

namespace simpla { namespace mesh
{


class MeshBase : public base::Object
{
    MeshBlockId m_id_;
    int m_level_;
    unsigned long m_status_flag_ = 0;
public:

    SP_OBJECT_HEAD(mesh::MeshBase, base::Object);

    MeshBase() { }

    ~MeshBase() { }

    MeshBlockId const &id() const { return m_id_; }

    int level() const { return m_level_; }

    virtual box_type box() const = 0;

    virtual MeshEntityRange range(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t size(MeshEntityType entityType = VERTEX) const = 0;

    virtual size_t hash(MeshEntityId const &) const = 0;

};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHBASE_H
