//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

/**
 *  Curvilinear
 *  metric is not uniform
 */
struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

    template <typename... Args>
    explicit SMesh(Args &&... args) : StructuredMesh(std::forward<Args>(args)...){};
    ~SMesh() override = default;

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME("SMesh");
    void InitializeData(Real time_now) override;
    void SetBoundaryCondition(Real time_now, Real time_dt) override;

    point_type point(EntityId s) const override;

    point_type local_coordinates(EntityId s, point_type const &r) const;

#define DECLARE_FIELD(_IFORM_, _DOF_, _NAME_, ...) \
    Field<this_type, Real, _IFORM_, _DOF_> _NAME_{this, "name"_ = __STRING(_NAME_), __VA_ARGS__};

    DECLARE_FIELD(VERTEX, 3, m_coordinates_, "COORDINATES"_);

    DECLARE_FIELD(VERTEX, 3, m_vertices_);

    DECLARE_FIELD(VERTEX, 1, m_vertex_volume_);
    DECLARE_FIELD(VERTEX, 1, m_vertex_inv_volume_);
    DECLARE_FIELD(VERTEX, 1, m_vertex_dual_volume_);
    DECLARE_FIELD(VERTEX, 1, m_vertex_inv_dual_volume_);

    DECLARE_FIELD(VOLUME, 1, m_volume_volume_);
    DECLARE_FIELD(VOLUME, 1, m_volume_inv_volume_);
    DECLARE_FIELD(VOLUME, 1, m_volume_dual_volume_);
    DECLARE_FIELD(VOLUME, 1, m_volume_inv_dual_volume_);

    DECLARE_FIELD(EDGE, 1, m_edge_volume_);
    DECLARE_FIELD(EDGE, 1, m_edge_inv_volume_);
    DECLARE_FIELD(EDGE, 1, m_edge_dual_volume_);
    DECLARE_FIELD(EDGE, 1, m_edge_inv_dual_volume_);

    DECLARE_FIELD(FACE, 1, m_face_volume_);
    DECLARE_FIELD(FACE, 1, m_face_inv_volume_);
    DECLARE_FIELD(FACE, 1, m_face_dual_volume_);
    DECLARE_FIELD(FACE, 1, m_face_inv_dual_volume_);

    DECLARE_FIELD(VERTEX, 1, m_vertex_hodge_);
    DECLARE_FIELD(EDGE, 1, m_edge_hodge_);
    DECLARE_FIELD(FACE, 1, m_face_hodge_);
    DECLARE_FIELD(VOLUME, 1, m_volume_hodge_);
#undef DECLARE_FIELD
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
