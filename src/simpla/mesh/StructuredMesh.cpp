//
// Created by salmon on 17-4-25.
//
#include "StructuredMesh.h"
#include <simpla/algebra/all.h>
namespace simpla {
namespace mesh {
using namespace algebra;
void StructuredMesh::InitializeRange(std::shared_ptr<geometry::GeoObject> const& g, Range<EntityId> body[4],
                                     Range<EntityId> boundary[4]) {
    if (g->CheckOverlap(GetBox()) != 0) { return; }
    Field<this_type, int, VOLUME, 9> m_tags_{this};
    m_tags_.Clear();
    m_tags_[0].Foreach([&](index_tuple const& idx, int& v) {
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

    m_tags_[1] = m_tags_[0](I, J, K) | (m_tags_[0](I + 1, J, K) << 1);
    m_tags_[2] = m_tags_[0](I, J, K) | (m_tags_[0](I, J + 1, K) << 2);
    m_tags_[4] = m_tags_[0](I, J, K) | (m_tags_[0](I, J, K + 1) << 4);
    m_tags_[3] = m_tags_[0](I, J, K)               //
                 | (m_tags_[0](I + 1, J, K) << 1)  //
                 | (m_tags_[0](I, J + 1, K) << 2)  //
                 | (m_tags_[0](I + 1, J + 1, K) << 3);
    m_tags_[5] = m_tags_[0](I, J, K)               //
                 | (m_tags_[0](I + 1, J, K) << 1)  //
                 | (m_tags_[0](I, J, K + 1) << 4)  //
                 | (m_tags_[0](I + 1, J, K + 1) << 5);
    m_tags_[6] = m_tags_[0](I, J, K)               //
                 | (m_tags_[0](I, J + 1, K) << 2)  //
                 | (m_tags_[0](I, J, K + 1) << 4)  //
                 | (m_tags_[0](I, J + 1, K + 1) << 6);
    m_tags_[7] = (m_tags_[0](I, J, K))                 //
                 | (m_tags_[0](I + 1, J, K) << 1)      //
                 | (m_tags_[0](I, J + 1, K) << 2)      //
                 | (m_tags_[0](I + 1, J + 1, K) << 3)  //
                 | (m_tags_[0](I, J, K + 1) << 4)      //
                 | (m_tags_[0](I + 1, J, K + 1) << 5)  //
                 | (m_tags_[0](I, J + 1, K + 1) << 6)  //
                 | (m_tags_[0](I + 1, J + 1, K + 1) << 7);

    static const int out_tag[8] = {
        0b00000001,  // 0
        0b00000011,  // 1
        0b00000101,  // 2
        0b00001111,  // 3
        0b00001001,  // 4
        0b00110011,  // 5
        0b01010101,  // 6
        0b11111111   // 7
    };
    std::shared_ptr<UnorderedRange<EntityId>> body_range[4];
    std::shared_ptr<UnorderedRange<EntityId>> boundary_range[4];

    for (int i = 0; i < 8; ++i) {
        body_range[i] = std::make_shared<UnorderedRange<EntityId>>(i);
        boundary_range[i] = std::make_shared<UnorderedRange<EntityId>>(i);

        body[i].append(body_range[i]);
        boundary[i].append(boundary_range[i]);
    }

    // VOLUME center
    m_tags_[7].Foreach([&](index_tuple const& idx, int& v) {
        EntityId s{.w = static_cast<int16_t>(7),
                   .x = static_cast<int16_t>(idx[0]),
                   .y = static_cast<int16_t>(idx[1]),
                   .z = static_cast<int16_t>(idx[2])};
        if (v == 0) {
            body_range[7]->Insert(s);
        } else {
            boundary_range[7]->Insert(s);
        }
    });

    m_tags_[7].Foreach([&](index_tuple const& idx, int& v) {
        //        EntityId s{.w = static_cast<int16_t>(0),
        //                   .x = static_cast<int16_t>(x),
        //                   .y = static_cast<int16_t>(y),
        //                   .z = static_cast<int16_t>(z)};
        //        int t7 = m_tags_[7](x, y, z);
        //        if (t7 == 0) {
        //            s.w = 7;
        //            in_range[7]->data().insert(s);
        //        } else if (t7 != out_tag[7]) {
        //            s.w = 7;
        //            brd_range[7]->data().insert(s);
        //            for (int w = 1; w < 6; ++w) {
        //                s.w = static_cast<int16_t>(w);
        //                auto t = m_tags_[w](x, y, z);
        //                if (t == 0) {
        //                    in_range[w]->Insert(s);
        //                } else if (t == out_tag[w]) {
        //                    brd_range[w]->Insert(s);
        //                }
        //            }
        //        }
    });
}
}  // namespace mesh{
}  // namespace simpla{
