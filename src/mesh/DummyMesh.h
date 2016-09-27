/** 
 * @file DummyMesh.h
 * @author salmon
 * @date 16-5-25 - 上午8:25
 *  */

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H

#include <memory>

#include "MeshCommon.h"
#include "MeshBase.h"
#include "MeshEntityRange.h"

namespace simpla { namespace mesh
{
struct DummyMesh : public MeshBase
{
    SP_OBJECT_HEAD(DummyMesh, MeshBase);

    typedef DummyMesh mesh_type;

    std::set<MeshEntityId> m_entities_;
    std::vector<point_type> m_points_;
    box_type m_box_{{0, 0, 0},
                    {1, 1, 1}};

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual box_type box() const { return m_box_; };

    virtual MeshEntityRange range(MeshEntityType) const
    {
        return MeshEntityRange(m_entities_.begin(), m_entities_.end());
    }

    virtual size_type size(MeshEntityType entityType = VERTEX) const { max_hash(entityType); };

    virtual size_type max_hash(MeshEntityType entityType = VERTEX) const { return m_points_.size(); }

    virtual size_type hash(MeshEntityId const &s) const { return static_cast<size_type>(s); }

    virtual point_type point(MeshEntityId const &s) const { return m_points_[hash(s)]; }

    virtual int get_adjacent_entities(MeshEntityId const &s, MeshEntityType t,
                                      MeshEntityId *p = nullptr) const
    {
        if (p != nullptr) { p[0] = s; }
        return 1;
    };


    virtual std::shared_ptr<MeshBase> refine(box_type const &b, int flag = 0) const
    {
        return std::dynamic_pointer_cast<MeshBase>(std::make_shared<DummyMesh>());
    };

    struct calculus_policy
    {
        template<typename TF, typename ...Args>
        static traits::value_type_t<TF> eval(mesh_type const &, TF const &,
                                             Args &&...args) { return traits::value_type_t<TF>(); }
    };
};

}}//namespace simpla { namespace get_mesh

#endif //SIMPLA_DUMMYMESH_H
