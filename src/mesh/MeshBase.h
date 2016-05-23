//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include "Mesh.h"

namespace simpla { namespace mesh
{
class EntityRange;

class MeshBase
{
    uuid m_id_;
public:

    uuid const &id() const { return m_id_; }

    virtual box_type box() const;

    virtual EntityRange range() const;

    virtual size_t size(EntityType entityType = VERTEX) const;

    virtual size_t hash(EntityId const &) const;

    virtual bool is_a(std::typeinfo const &t_info) const = 0;

    template<typename T> inline bool is_a() const
    {
        return (std::is_base_of<MeshBase, T>::value && is_a(typeid(T)));
    };

};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHBASE_H
