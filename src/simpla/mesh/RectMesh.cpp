//
// Created by salmon on 17-6-1.
//

#include "RectMesh.h"

#include <simpla/utilities/EntityIdCoder.h>
namespace simpla {
namespace mesh {
REGISTER_CREATOR(RectMesh);

void RectMesh::InitializeData(Real time_now) {
    StructuredMesh::InitializeData(time_now);
    m_coordinates_.Clear();
    m_coordinates_ = [&](EntityId s) -> point_type { return global_coordinates(s); };

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
        return chart->box_volume(point(EntityId{.w = 0b111,
                                                .x = static_cast<int16_t>(s.x - 1),
                                                .y = static_cast<int16_t>(s.y - 1),
                                                .z = static_cast<int16_t>(s.z - 1)}),
                                 point(EntityId{.w = 0b111, .x = s.x, .y = s.y, .z = s.z}));
    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](EntityId s) -> Real {
        return chart->box_volume(point(EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z}),
                                 point(EntityId{.w = 0b0,
                                                .x = static_cast<int16_t>(s.x + 1),
                                                .y = static_cast<int16_t>(s.y + 1),
                                                .z = static_cast<int16_t>(s.z + 1)}));
    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z}),
                             point(EntityId{.w = 0b0,
                                            .x = static_cast<int16_t>(s.x + (s.w & 0b111) == 0b001 ? 1 : 0),
                                            .y = static_cast<int16_t>(s.y + (s.w & 0b111) == 0b010 ? 1 : 0),
                                            .z = static_cast<int16_t>(s.z + (s.w & 0b111) == 0b100 ? 1 : 0)}));
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](EntityId s) -> Real {
        return chart->box_area(point(EntityId{.w = 0b111,
                                              .x = static_cast<int16_t>(s.x - ((s.w & 0b111) != 0b001 ? 1 : 0)),
                                              .y = static_cast<int16_t>(s.y - ((s.w & 0b111) != 0b010 ? 1 : 0)),
                                              .z = static_cast<int16_t>(s.z - ((s.w & 0b111) != 0b100 ? 1 : 0))}),
                               point(EntityId{.w = 0b111, .x = s.x, .y = s.y, .z = s.z}));
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Real {
        return chart->box_area(point(EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z}),
                               point(EntityId{.w = 0b0,
                                              .x = static_cast<int16_t>(s.x + ((s.w & 0b111) != 0b110 ? 1 : 0)),
                                              .y = static_cast<int16_t>(s.y + ((s.w & 0b111) != 0b101 ? 1 : 0)),
                                              .z = static_cast<int16_t>(s.z + ((s.w & 0b111) != 0b011 ? 1 : 0))}));

    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;
    m_face_dual_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{.w = 0b111,
                                            .x = static_cast<int16_t>(s.x - (s.w & 0b111) == 0b110 ? 1 : 0),
                                            .y = static_cast<int16_t>(s.y - (s.w & 0b111) == 0b101 ? 1 : 0),
                                            .z = static_cast<int16_t>(s.z - (s.w & 0b111) == 0b011 ? 1 : 0)}),
                             point(EntityId{.w = 0b111, .x = s.x, .y = s.y, .z = s.z}));
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

void RectMesh::SetBoundaryCondition(Real time_now, Real time_dt) {
    StructuredMesh::SetBoundaryCondition(time_now, time_dt);

    m_vertex_volume_[GetRange("VERTEX_PATCH_BOUNDARY")] = 0;
    m_vertex_dual_volume_[GetRange("VERTEX_PATCH_BOUNDARY")] = 0;

    m_edge_volume_[GetRange("EDGE_PATCH_BOUNDARY")] = 0;
    m_edge_dual_volume_[GetRange("EDGE_PATCH_BOUNDARY")] = 0;

    m_face_volume_[GetRange("FACE_PATCH_BOUNDARY")] = 0;
    m_face_dual_volume_[GetRange("FACE_PATCH_BOUNDARY")] = 0;

    m_volume_volume_[GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
    m_volume_dual_volume_[GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
}
}  // namespace mesh
}  // namespace simpla