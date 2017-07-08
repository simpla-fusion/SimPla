//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "EBMesh.h"
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

    EBMesh<SMesh> boundary{this};

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME(SMesh);
    void InitializeData(Real time_now) override;
    void SetBoundaryCondition(Real time_now, Real time_dt) override;

    point_type local_coordinates(entity_id_type s, Real const *r = nullptr) const override;

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
    Field<this_type, Real, VERTEX> m_vertex_hodge_{this, "name"_ = "m_vertex_hodge_"};
    Field<this_type, Real, EDGE> m_edge_hodge_{this, "name"_ = "m_edge_hodge_"};
    Field<this_type, Real, FACE> m_face_hodge_{this, "name"_ = "m_face_hodge_"};
    Field<this_type, Real, VOLUME> m_volume_hodge_{this, "name"_ = "m_volume_hodge_"};
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
