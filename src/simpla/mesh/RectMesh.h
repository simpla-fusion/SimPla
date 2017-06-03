//
// Created by salmon on 17-6-1.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

/**
 * Axis are perpendicular
 */
struct RectMesh : public StructuredMesh, public AttributeGroup {
   public:
    SP_OBJECT_HEAD(RectMesh, StructuredMesh)

    template <typename... Args>
    explicit RectMesh(Args &&... args) : StructuredMesh(std::forward<Args>(args)...){};
    ~RectMesh() override = default;

    SP_DEFAULT_CONSTRUCT(RectMesh)
    DECLARE_REGISTER_NAME("RectMesh");

    void InitializeData(Real time_now) override;
    void SetBoundaryCondition(Real time_now, Real time_dt) override;
    void Push(Patch *) override;
    void Pop(Patch *) override;
#define DECLARE_FIELD(_IFORM_, _DOF_, _NAME_, ...) \
    Field<this_type, Real, _IFORM_, _DOF_> _NAME_{this, "name"_ = __STRING(_NAME_), __VA_ARGS__};

    DECLARE_FIELD(VERTEX, 3, m_coordinates_, "COORDINATES"_);

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
#endif  // SIMPLA_RECTMESH_H
