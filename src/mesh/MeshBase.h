//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <typeinfo>
#include "Mesh.h"
#include "MeshEntity.h"

namespace simpla { namespace mesh
{


class MeshBase
{
    MeshBlockId m_id_;
    int m_level_;
    unsigned long m_status_flag_ = 0;
public:

    MeshBlockId const &id() const { return m_id_; }

    int level() const { return m_level_; }

    virtual box_type box() const;

    virtual EntityRange range() const;

    virtual size_t size(MeshEntityType entityType = VERTEX) const;

    virtual size_t hash(EntityId const &) const;

    virtual bool is_a(std::type_info const &t_info) const = 0;

    template<typename T> inline bool is_a() const
    {
        return (std::is_base_of<MeshBase, T>::value && is_a(typeid(T)));
    };

    bool is_local() const { return (m_status_flag_ & 1) == 0; }
};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHBASE_H
