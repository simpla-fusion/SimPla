//
// Created by salmon on 17-4-25.
//
#include "StructuredMesh.h"
#include <simpla/algebra/all.h>
#include <simpla/utilities/EntityId.h>
namespace simpla {
namespace mesh {
using namespace algebra;

void StructuredMesh::RegisterRanges(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix,
                                    std::map<std::string, EntityRange> &ranges) {
    if (g == nullptr) { return; }
    auto overlap = g->CheckOverlap(GetBox());
    if (overlap == 1) { return; }
    if (overlap == -1) {
        ranges[prefix + ".VERTEX_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(0), 0));

        ranges[prefix + ".EDGE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(1), 1));
        ranges[prefix + ".EDGE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(2), 2));
        ranges[prefix + ".EDGE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(4), 4));

        ranges[prefix + ".FACE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(3), 3));
        ranges[prefix + ".FACE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(5), 5));
        ranges[prefix + ".FACE_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(6), 6));

        ranges[prefix + ".VOLUME_BODY"].append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(7), 7));

        return;
    }

    vertex_tags.Clear();

    //    vertex_tags[0].Foreach([&](index_tuple const &idx, int &v) {
    //        if (g->CheckInside(point(idx[0], idx[1], idx[2])) == 0) { v = 1; }
    //    });

    index_tuple ib, ie;
    std::tie(ib, ie) = GetBlock()->GetIndexBox();
    ie += 1;

    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                EntityId s = {.w = 0,
                              .x = static_cast<int16_t>(I),  //
                              .y = static_cast<int16_t>(J),  //
                              .z = static_cast<int16_t>(K)};
                if (g->CheckInside(point(I, J, K)) == 0) { vertex_tags[0](I, J, K) = 1; }
            }

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

    std::tie(ib, ie) = GetBlock()->GetIndexBox();

    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                EntityId s = {.w = 0,                        //
                              .x = static_cast<int16_t>(I),  //
                              .y = static_cast<int16_t>(J),  //
                              .z = static_cast<int16_t>(K)};

                int volume_tags = (vertex_tags[s0 + s] << 0) |  //
                                  (vertex_tags[s1 + s] << 1) |  //
                                  (vertex_tags[s2 + s] << 2) |  //
                                  (vertex_tags[s3 + s] << 3) |  //
                                  (vertex_tags[s4 + s] << 4) |  //
                                  (vertex_tags[s5 + s] << 5) |  //
                                  (vertex_tags[s6 + s] << 6) |  //
                                  (vertex_tags[s7 + s] << 7);
                //
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

                    VERTEX_body->Insert(s0 + s);
                    VERTEX_body->Insert(s1 + s);
                    VERTEX_body->Insert(s2 + s);
                    VERTEX_body->Insert(s3 + s);
                    VERTEX_body->Insert(s4 + s);
                    VERTEX_body->Insert(s5 + s);
                    VERTEX_body->Insert(s6 + s);
                    VERTEX_body->Insert(s7 + s);

                    EDGE_body->Insert((t1 | s0) + s);
                    EDGE_body->Insert((t1 | s2) + s);
                    EDGE_body->Insert((t1 | s4) + s);
                    EDGE_body->Insert((t1 | s6) + s);

                    EDGE_body->Insert((t2 | s0) + s);
                    EDGE_body->Insert((t2 | s1) + s);
                    EDGE_body->Insert((t2 | s4) + s);
                    EDGE_body->Insert((t2 | s5) + s);

                    EDGE_body->Insert((t4 | s0) + s);
                    EDGE_body->Insert((t4 | s1) + s);
                    EDGE_body->Insert((t4 | s2) + s);
                    EDGE_body->Insert((t4 | s3) + s);

                    FACE_body->Insert((t3 | s0) + s);
                    FACE_body->Insert((t5 | s0) + s);
                    FACE_body->Insert((t6 | s0) + s);
                    FACE_body->Insert((t6 | s1) + s);
                    FACE_body->Insert((t5 | s2) + s);
                    FACE_body->Insert((t3 | s4) + s);

                    VOLUME_body->Insert(t7 | s0);

                } else if (volume_tags < 0b11111111) {
                    if ((volume_tags & b0) != 0) { VERTEX_boundary->Insert(s0 + s); }
                    if ((volume_tags & b1) != 0) { VERTEX_boundary->Insert(s1 + s); }
                    if ((volume_tags & b2) != 0) { VERTEX_boundary->Insert(s2 + s); }
                    if ((volume_tags & b3) != 0) { VERTEX_boundary->Insert(s3 + s); }
                    if ((volume_tags & b4) != 0) { VERTEX_boundary->Insert(s4 + s); }
                    if ((volume_tags & b5) != 0) { VERTEX_boundary->Insert(s5 + s); }
                    if ((volume_tags & b6) != 0) { VERTEX_boundary->Insert(s6 + s); }
                    if ((volume_tags & b7) != 0) { VERTEX_boundary->Insert(s7 + s); }

#define CHECK_TAG(_TAG_) if ((volume_tags & _TAG_) == _TAG_)
                    CHECK_TAG(0b00000011) { EDGE_PARA_boundary->Insert(t1 | s0 + s); }
                    CHECK_TAG(0b00001100) { EDGE_PARA_boundary->Insert(t1 | s2 + s); }
                    CHECK_TAG(0b00110000) { EDGE_PARA_boundary->Insert(t1 | s4 + s); }
                    CHECK_TAG(0b11000000) { EDGE_PARA_boundary->Insert(t1 | s6 + s); }
                    CHECK_TAG(0b00000101) { EDGE_PARA_boundary->Insert(t2 | s0 + s); }
                    CHECK_TAG(0b00001010) { EDGE_PARA_boundary->Insert(t2 | s1 + s); }
                    CHECK_TAG(0b01010000) { EDGE_PARA_boundary->Insert(t2 | s4 + s); }
                    CHECK_TAG(0b10100000) { EDGE_PARA_boundary->Insert(t2 | s5 + s); }
                    CHECK_TAG(0b00010001) { EDGE_PARA_boundary->Insert(t4 | s0 + s); }
                    CHECK_TAG(0b00100010) { EDGE_PARA_boundary->Insert(t4 | s1 + s); }
                    CHECK_TAG(0b01000100) { EDGE_PARA_boundary->Insert(t4 | s2 + s); }
                    CHECK_TAG(0b10001000) { EDGE_PARA_boundary->Insert(t4 | s3 + s); }

//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t3 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t5 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t6 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t6 | s1 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t5 | s2 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t3 | s4 + s); }
#undef CHECK_TAG
                    VOLUME_boundary->Insert(t7 + s);
                }
            }
    ranges[prefix + ".VERTEX_BODY"].append(VERTEX_body);
    ranges[prefix + ".EDGE_BODY"].append(EDGE_body);
    ranges[prefix + ".FACE_BODY"].append(FACE_body);
    ranges[prefix + ".VOLUME_BODY"].append(VOLUME_body);
    ranges[prefix + ".VERTEX_BOUNDARY"].append(VERTEX_boundary);
    ranges[prefix + ".EDGE_PARA_BOUNDARY"].append(EDGE_PARA_boundary);
    ranges[prefix + ".FACE_PARA_BOUNDARY"].append(FACE_PARA_boundary);
    ranges[prefix + ".EDGE_PERP_BOUNDARY"].append(EDGE_PERP_boundary);
    ranges[prefix + ".FACE_PERP_BOUNDARY"].append(FACE_PERP_boundary);
    ranges[prefix + ".VOLUME_BOUNDARY"].append(VOLUME_boundary);
}

}  // namespace mesh{
}  // namespace simpla{
