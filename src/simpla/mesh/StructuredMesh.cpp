//
// Created by salmon on 17-4-25.
//
#include "StructuredMesh.h"
#include <simpla/algebra/all.h>
namespace simpla {
namespace mesh {
using namespace algebra;

void StructuredMesh::InitializeRange(std::shared_ptr<geometry::GeoObject> const& g, EntityRange* range) {
    auto overlap = g->CheckOverlap(GetBox());
    if (overlap == 1) { return; }
    if (overlap == -1) {
        auto idx_box = GetBlock()->GetIndexBox();
        {
            index_tuple ib, ie;
            std::tie(ib, ie) = idx_box;
            ie += 1;
            range[VERTEX_BODY].append(std::make_shared<ContinueRange<EntityId>>(ib, ie, 0));
        }

        {
            index_tuple ib, ie;
            for (int i = 0; i < 3; ++i) {
                std::tie(ib, ie) = idx_box;
                ie[(i + 1) % 3] += 1;
                ie[(i + 2) % 3] += 1;
                range[EDGE_BODY].append(std::make_shared<ContinueRange<EntityId>>(ib, ie, 0b1 << i));
            }
        }

        {
            index_tuple ib, ie;
            for (int i = 0; i < 3; ++i) {
                std::tie(ib, ie) = idx_box;
                ie[i] += 1;
                range[FACE_BODY].append(std::make_shared<ContinueRange<EntityId>>(ib, ie, (~(0b1 << i)) & 0b111));
            }
        }
        {
            index_tuple ib, ie;
            std::tie(ib, ie) = idx_box;
            range[VOLUME_BODY].append(std::make_shared<ContinueRange<EntityId>>(ib, ie, 0b111));
        }
        return;
    }

    Field<this_type, int, VERTEX> vertex_tags{this};

    vertex_tags.Clear();
    vertex_tags[0].Foreach([&](index_tuple const& idx, int& v) {
        if (!g->CheckInside(point(idx[0], idx[0], idx[0]))) { v = 1; }
    });
    /**
    *\verbatim
    *                ^s (dl)
    *               /
    *   (dz) t     /
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
    *       000-------------001---> r (dr)
    *
    *\endverbatim
    */

    auto VERTEX_body = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_body = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_body = std::make_shared<UnorderedRange<EntityId>>();
    auto VOLUME_body = std::make_shared<UnorderedRange<EntityId>>();

    auto VERTEX_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_PARA_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_PARA_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_PERP_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_PERP_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto VOLUME_boundary = std::make_shared<UnorderedRange<EntityId>>();

    static const int b0 = 0b000;
    static const int b1 = 0b001;
    static const int b2 = 0b010;
    static const int b3 = 0b011;
    static const int b4 = 0b100;
    static const int b5 = 0b101;
    static const int b6 = 0b110;
    static const int b7 = 0b111;
    static const EntityId t0 = {.w = 0b000, .x = 0, .y = 0, .z = 0};
    static const EntityId t1 = {.w = 0b001, .x = 0, .y = 0, .z = 0};
    static const EntityId t2 = {.w = 0b010, .x = 0, .y = 0, .z = 0};
    static const EntityId t3 = {.w = 0b011, .x = 0, .y = 0, .z = 0};
    static const EntityId t4 = {.w = 0b100, .x = 0, .y = 0, .z = 0};
    static const EntityId t5 = {.w = 0b101, .x = 0, .y = 0, .z = 0};
    static const EntityId t6 = {.w = 0b110, .x = 0, .y = 0, .z = 0};
    static const EntityId t7 = {.w = 0b111, .x = 0, .y = 0, .z = 0};

    static const EntityId s0 = {.w = 0, .x = 0, .y = 0, .z = 0};
    static const EntityId s1 = {.w = 0, .x = 1, .y = 0, .z = 0};
    static const EntityId s2 = {.w = 0, .x = 0, .y = 1, .z = 0};
    static const EntityId s3 = {.w = 0, .x = 1, .y = 1, .z = 0};
    static const EntityId s4 = {.w = 0, .x = 0, .y = 0, .z = 1};
    static const EntityId s5 = {.w = 0, .x = 1, .y = 0, .z = 1};
    static const EntityId s6 = {.w = 0, .x = 0, .y = 1, .z = 1};
    static const EntityId s7 = {.w = 0, .x = 1, .y = 1, .z = 1};

    index_tuple ib, ie;
    std::tie(ib, ie) = GetBlock()->GetIndexBox();

    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                EntityId s = {.w = 0,                        //
                              .x = static_cast<int16_t>(I),  //
                              .y = static_cast<int16_t>(J),  //
                              .z = static_cast<int16_t>(K)};

                int volume_tags = (vertex_tags[s + s0] << 0) |  //
                                  (vertex_tags[s + s1] << 1) |  //
                                  (vertex_tags[s + s2] << 2) |  //
                                  (vertex_tags[s + s3] << 3) |  //
                                  (vertex_tags[s + s4] << 4) |  //
                                  (vertex_tags[s + s5] << 5) |  //
                                  (vertex_tags[s + s6] << 6) |  //
                                  (vertex_tags[s + s7] << 7);

                if (volume_tags == 0) {
                    /**
                     *\verbatim
                     *                ^s (dl)
                     *               /
                     *   (dz) t     /
                     *        ^    /
                     *        |   6 --------------7
                     *        |  /|              /|
                     *        | / |             / |
                     *        |/  |            /  |
                     *        4 --|---------- 5   |
                     *        | m |           |   |
                     *        |   2 ----------|-- 3
                     *        |  /            |  /
                     *        | /             | /
                     *        |/              |/
                     *        0 ------------- 1---> r (dr)
                     *
                     *\endverbatim
                     */

                    VERTEX_body->Insert(s + s0);
                    VERTEX_body->Insert(s + s1);
                    VERTEX_body->Insert(s + s2);
                    VERTEX_body->Insert(s + s3);
                    VERTEX_body->Insert(s + s4);
                    VERTEX_body->Insert(s + s5);
                    VERTEX_body->Insert(s + s6);
                    VERTEX_body->Insert(s + s7);

                    EDGE_body->Insert(s + (t1 | s0));
                    EDGE_body->Insert(s + (t1 | s2));
                    EDGE_body->Insert(s + (t1 | s4));
                    EDGE_body->Insert(s + (t1 | s6));
                    EDGE_body->Insert(s + (t2 | s0));
                    EDGE_body->Insert(s + (t2 | s1));
                    EDGE_body->Insert(s + (t2 | s4));
                    EDGE_body->Insert(s + (t2 | s5));
                    EDGE_body->Insert(s + (t4 | s0));
                    EDGE_body->Insert(s + (t4 | s1));
                    EDGE_body->Insert(s + (t4 | s2));
                    EDGE_body->Insert(s + (t4 | s3));
                    FACE_body->Insert(s + (t3 | s0));
                    FACE_body->Insert(s + (t5 | s0));
                    FACE_body->Insert(s + (t6 | s0));
                    FACE_body->Insert(s + (t6 | s1));
                    FACE_body->Insert(s + (t5 | s2));
                    FACE_body->Insert(s + (t3 | s4));

                    VOLUME_body->Insert(t7 | s0);

                } else if (volume_tags < 0b11111111) {
                    if ((volume_tags & b0) != 0) { VERTEX_boundary->Insert(s + s0); }
                    if ((volume_tags & b1) != 0) { VERTEX_boundary->Insert(s + s1); }
                    if ((volume_tags & b2) != 0) { VERTEX_boundary->Insert(s + s2); }
                    if ((volume_tags & b3) != 0) { VERTEX_boundary->Insert(s + s3); }
                    if ((volume_tags & b4) != 0) { VERTEX_boundary->Insert(s + s4); }
                    if ((volume_tags & b5) != 0) { VERTEX_boundary->Insert(s + s5); }
                    if ((volume_tags & b6) != 0) { VERTEX_boundary->Insert(s + s6); }
                    if ((volume_tags & b7) != 0) { VERTEX_boundary->Insert(s + s7); }

                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t1 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t1 | s2); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t1 | s4); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t1 | s6); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t2 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t2 | s1); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t2 | s4); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t2 | s5); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t4 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t4 | s1); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t4 | s2); }
                    if ((volume_tags & (0b1 << 0)) != 0) { EDGE_PARA_boundary->Insert(s + t4 | s3); }

                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t3 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t5 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t6 | s0); }
                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t6 | s1); }
                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t5 | s2); }
                    if ((volume_tags & (0b1 << 0)) != 0) { FACE_PARA_boundary->Insert(s + t3 | s4); }

                    VOLUME_boundary->Insert(s + t7 | s0);
                }
            }

    range[VERTEX_BODY].append(VERTEX_body);
    range[EDGE_BODY].append(EDGE_body);
    range[FACE_BODY].append(FACE_body);
    range[VOLUME_BODY].append(VOLUME_body);
    range[VERTEX_BOUNDARY].append(VERTEX_boundary);
    range[EDGE_PARA_BOUNDARY].append(EDGE_PARA_boundary);
    range[FACE_PARA_BOUNDARY].append(FACE_PARA_boundary);
    range[EDGE_PERP_BOUNDARY].append(EDGE_PERP_boundary);
    range[FACE_PERP_BOUNDARY].append(FACE_PERP_boundary);
    range[VOLUME_BOUNDARY].append(VOLUME_boundary);
}

}  // namespace mesh{
}  // namespace simpla{
