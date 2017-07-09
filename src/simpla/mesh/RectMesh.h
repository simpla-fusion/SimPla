//
// Created by salmon on 17-6-1.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H
#include <simpla/algebra/CalculusPolicy.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/Domain.h>
#include "Mesh.h"
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

using namespace simpla::data;
/**
 * Axis are perpendicular
 */
struct RectMesh : public engine::Domain, public StructuredMesh {
   public:
    SP_OBJECT_HEAD(RectMesh, engine::Domain)

    template <typename... Args>
    explicit RectMesh(Args &&... args) : engine::Domain(std::forward<Args>(args)...){};
    ~RectMesh() override = default;

    SP_DEFAULT_CONSTRUCT(RectMesh)
    DECLARE_REGISTER_NAME(RectMesh);

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real time_dt) override;

    Field<this_type, Real, VERTEX, 3> m_coordinates_{this, "name"_ = "m_coordinates_" /*, "COORDINATES"_*/};
    Field<this_type, Real, VERTEX, 3> m_vertices_{this, "name"_ = "m_vertices_"};

    Field<this_type, Real, VERTEX> m_vertex_volume_{this, "name"_ = "m_vertex_volume_"};
    Field<this_type, Real, VERTEX> m_vertex_inv_volume_{this, "name"_ = "m_vertex_inv_volume_"};
    Field<this_type, Real, VERTEX> m_vertex_dual_volume_{this, "name"_ = "m_vertex_dual_volume_"};
    Field<this_type, Real, VERTEX> m_vertex_inv_dual_volume_{this, "name"_ = "m_vertex_inv_dual_volume_"};
    Field<this_type, Real, VOLUME> m_volume_volume_{this, "name"_ = "m_volume_volume_"};
    Field<this_type, Real, VOLUME> m_volume_inv_volume_{this, "name"_ = "m_volume_inv_volume_"};
    Field<this_type, Real, VOLUME> m_volume_dual_volume_{this, "name"_ = "m_volume_dual_volume_"};
    Field<this_type, Real, VOLUME> m_volume_inv_dual_volume_{this, "name"_ = "m_volume_inv_dual_volume_"};
    Field<this_type, Real, EDGE> m_edge_volume_{this, "name"_ = "m_edge_volume_"};
    Field<this_type, Real, EDGE> m_edge_inv_volume_{this, "name"_ = "m_edge_inv_volume_"};
    Field<this_type, Real, EDGE> m_edge_dual_volume_{this, "name"_ = "m_edge_dual_volume_"};
    Field<this_type, Real, EDGE> m_edge_inv_dual_volume_{this, "name"_ = "m_edge_inv_dual_volume_"};
    Field<this_type, Real, FACE> m_face_volume_{this, "name"_ = "m_face_volume_"};
    Field<this_type, Real, FACE> m_face_inv_volume_{this, "name"_ = "m_face_inv_volume_"};
    Field<this_type, Real, FACE> m_face_dual_volume_{this, "name"_ = "m_face_dual_volume_"};
    Field<this_type, Real, FACE> m_face_inv_dual_volume_{this, "name"_ = "m_face_inv_dual_volume_"};
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_RECTMESH_H
