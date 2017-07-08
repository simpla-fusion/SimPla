//
// Created by salmon on 17-6-1.
//
#include "RectMesh.h"
#include <simpla/SIMPLA_config.h>

namespace simpla {
namespace mesh {
REGISTER_CREATOR(RectMesh);

void RectMesh::InitializeData(Real time_now) {
    StructuredMesh::InitializeData(time_now);

    m_coordinates_ = [&](EntityId s) -> point_type { return global_coordinates(s); };
    m_vertices_ = [&](EntityId s) -> point_type { return global_coordinates(s); };
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

    /**
        *\verbatim
        *                ^y (dl)
        *               /
        *   (dz) z     /
        *        ^    /
        *        |  110-------------111
        *        |  /|              /|
        *        | / |             / |
        *        |/  |            /  |
        *       100--|----------101  |
        *        | m |           |   |
        *        |  010----------|--011
        *        |  /            |  /
        *        | /             | /
        *        |/              |/
        *       000-------------001---> x (dr)
        *
        *\endverbatim
        */

    auto chart = GetChart();
    m_vertex_volume_ = 1.0;
    m_vertex_inv_volume_ = 1.0;
    m_vertex_dual_volume_ = [&](EntityId s) -> Real {
        return chart->volume(point(EntityId{static_cast<int16_t>(s.x - 1), static_cast<int16_t>(s.y - 1),
                                            static_cast<int16_t>(s.z - 1), 0b111}),
                             point(EntityId{s.x, s.y, s.z, 0b111}));
    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](EntityId s) -> Real {
        return chart->volume(point(EntityId{s.x, s.y, s.z, 0b0}),
                             point(EntityId{static_cast<int16_t>(s.x + 1), static_cast<int16_t>(s.y + 1),
                                            static_cast<int16_t>(s.z + 1), 0b0}));
    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{s.x, s.y, s.z, 0b0}),
                             point(EntityId{static_cast<int16_t>(s.x + ((s.w & 0b111) == 0b001 ? 1 : 0)),
                                            static_cast<int16_t>(s.y + ((s.w & 0b111) == 0b010 ? 1 : 0)),
                                            static_cast<int16_t>(s.z + ((s.w & 0b111) == 0b100 ? 1 : 0)), 0b0}),
                             EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](EntityId s) -> Real {
        return chart->area(point(EntityId{static_cast<int16_t>(s.x - ((s.w & 0b111) != 0b001 ? 1 : 0)),
                                          static_cast<int16_t>(s.y - ((s.w & 0b111) != 0b010 ? 1 : 0)),
                                          static_cast<int16_t>(s.z - ((s.w & 0b111) != 0b100 ? 1 : 0)), 0b111}),
                           point(EntityId{s.x, s.y, s.z, 0b111}), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Real {
        return chart->area(point(EntityId{s.x, s.y, s.z, 0b0}),
                           point(EntityId{static_cast<int16_t>(s.x + ((s.w & 0b111) != 0b110 ? 1 : 0)),
                                          static_cast<int16_t>(s.y + ((s.w & 0b111) != 0b101 ? 1 : 0)),
                                          static_cast<int16_t>(s.z + ((s.w & 0b111) != 0b011 ? 1 : 0)), 0b0}),
                           EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);

    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;

    m_face_dual_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{static_cast<int16_t>(s.x - ((s.w & 0b111) == 0b110 ? 1 : 0)),
                                            static_cast<int16_t>(s.y - ((s.w & 0b111) == 0b101 ? 1 : 0)),
                                            static_cast<int16_t>(s.z - ((s.w & 0b111) == 0b011 ? 1 : 0)), 0b111}),
                             point(EntityId{s.x, s.y, s.z, 0b111}), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;

    SetBoundaryCondition(0, 0);
};

void RectMesh::SetBoundaryCondition(Real time_now, Real time_dt) {
    StructuredMesh::SetBoundaryCondition(time_now, time_dt);

    boundary.Fill(m_vertex_volume_, 0);
    boundary.Fill(m_vertex_dual_volume_, 0);
    boundary.Fill(m_vertex_inv_volume_, 0);
    boundary.Fill(m_vertex_inv_dual_volume_, 0);

    boundary.Fill(m_edge_volume_, 0);
    boundary.Fill(m_edge_dual_volume_, 0);
    boundary.Fill(m_edge_inv_volume_, 0);
    boundary.Fill(m_edge_inv_dual_volume_, 0);

    boundary.Fill(m_face_volume_, 0);
    boundary.Fill(m_face_dual_volume_, 0);
    boundary.Fill(m_face_inv_volume_, 0);
    boundary.Fill(m_face_inv_dual_volume_, 0);

    boundary.Fill(m_volume_volume_, 0);
    boundary.Fill(m_volume_dual_volume_, 0);
    boundary.Fill(m_volume_inv_volume_, 0);
    boundary.Fill(m_volume_inv_dual_volume_, 0);
}
}  // namespace mesh
}  // namespace simpla