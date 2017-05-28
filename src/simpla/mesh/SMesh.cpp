//
// Created by salmon on 17-4-25.
//

#include "SMesh.h"
#include <simpla/utilities/EntityIdCoder.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

void SMesh::InitialCondition(Real time_now) {
    StructuredMesh::InitialCondition(time_now);

    m_vertices_.Clear();

    m_vertex_volume_.Clear();
    m_vertex_inv_volume_.Clear();
    m_vertex_dual_volume_.Clear();
    m_vertex_inv_dual_volume_.Clear();
    m_volume_volume_.Clear();
    m_volume_inv_volume_.Clear();
    m_volume_dual_volume_.Clear();
    m_volume_inv_dual_volume_.Clear();
    m_edge_volume_.Clear();
    m_edge_inv_volume_.Clear();
    m_edge_dual_volume_.Clear();
    m_edge_inv_dual_volume_.Clear();
    m_face_volume_.Clear();
    m_face_inv_volume_.Clear();
    m_face_dual_volume_.Clear();
    m_face_inv_dual_volume_.Clear();
};

void SMesh::BoundaryCondition(Real time_now, Real time_dt) {
    m_vertex_volume_[GetDomain()->GetRange("VERTEX_PATCH_BOUNDARY")] = 0;
    m_vertex_dual_volume_[GetDomain()->GetRange("VERTEX_PATCH_BOUNDARY")] = 0;

    m_edge_volume_[GetDomain()->GetRange("EDGE_PATCH_BOUNDARY")] = 0;
    m_edge_dual_volume_[GetDomain()->GetRange("EDGE_PATCH_BOUNDARY")] = 0;

    m_face_volume_[GetDomain()->GetRange("FACE_PATCH_BOUNDARY")] = 0;
    m_face_dual_volume_[GetDomain()->GetRange("FACE_PATCH_BOUNDARY")] = 0;

    m_volume_volume_[GetDomain()->GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
    m_volume_dual_volume_[GetDomain()->GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
}
}
}