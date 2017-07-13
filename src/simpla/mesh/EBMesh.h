//
// Created by salmon on 17-7-12.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H

#include <simpla/algebra/EntityId.h>
#include <simpla/algebra/algebra.h>
#include <simpla/data/data.h>
#include <simpla/engine/Domain.h>
#include <simpla/utilities/Range.h>

namespace simpla {
namespace mesh {

template <typename THost>
struct EBMesh {
    DOMAIN_POLICY_HEAD(EBMesh);

   public:
    void InitialCondition(Real time_now);
};

template <typename THost>
void EBMesh<THost>::InitialCondition(Real time_now) {
    if (m_host_->GetModel() == nullptr || m_host_->GetModel()->GetBoundary() == nullptr) { return; }

    auto g = m_host_->GetModel()->GetBoundary();

    Real ratio = g->CheckOverlap(m_host_->GetBox());

    if (ratio < EPSILON) {
        m_host_->GetRange("BODY_0")->append(nullptr);
        m_host_->GetRange("BODY_1")->append(nullptr);
        m_host_->GetRange("BODY_2")->append(nullptr);
        m_host_->GetRange("BODY_3")->append(nullptr);
        m_host_->GetRange("BOUNDARY_0")->append(nullptr);
        m_host_->GetRange("BOUNDARY_3")->append(nullptr);

        m_host_->GetRange("PARA_BOUNDARY_1")->append(nullptr);
        m_host_->GetRange("PARA_BOUNDARY_2")->append(nullptr);

        m_host_->GetRange("PERP_BOUNDARY_1")->append(nullptr);
        m_host_->GetRange("PERP_BOUNDARY_2")->append(nullptr);
        return;
    }
    //    if (1.0 - ratio < EPSILON) {  // all in
    //        // range["BODY_0"]->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(0),
    //        0));
    //        //
    //        //        range["BODY_1"]
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(1), 1))
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(2), 2))
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(4), 4));
    //        //
    //        //        range["BODY_2"]
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(3), 3))
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(5), 5))
    //        //            ->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(6), 6));
    //        //
    //        // range["BODY_3"]->append(std::make_shared<ContinueRange<EntityId>>(m_host_->GetIndexBox(7),
    //        7));
    //        return;
    //    }
    Field<host_type, int, VERTEX> vertex_tags{m_host_};

    vertex_tags = [&](point_type const &x) { return g->CheckInside(x) ? 1 : 0; };

    index_tuple ib, ie;
    std::tie(ib, ie) = m_host_->GetIndexBox(VERTEX);
    auto b = m_host_->GetBox();

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
    static const EntityId t0 = {0, 0, 0, 0b000};
    static const EntityId t1 = {0, 0, 0, 0b001};
    static const EntityId t2 = {0, 0, 0, 0b010};
    static const EntityId t3 = {0, 0, 0, 0b011};
    static const EntityId t4 = {0, 0, 0, 0b100};
    static const EntityId t5 = {0, 0, 0, 0b101};
    static const EntityId t6 = {0, 0, 0, 0b110};
    static const EntityId t7 = {0, 0, 0, 0b111};

    static const EntityId s0 = {0, 0, 0, 0};
    static const EntityId s1 = {1, 0, 0, 0};
    static const EntityId s2 = {0, 1, 0, 0};
    static const EntityId s3 = {1, 1, 0, 0};
    static const EntityId s4 = {0, 0, 1, 0};
    static const EntityId s5 = {1, 0, 1, 0};
    static const EntityId s6 = {0, 1, 1, 0};
    static const EntityId s7 = {1, 1, 1, 0};

    std::tie(ib, ie) = m_host_->GetIndexBox(VOLUME);

#pragma omp parallel for
    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                EntityId s = {static_cast<int16_t>(I), static_cast<int16_t>(J), static_cast<int16_t>(K), 0};

                int volume_tags = ((vertex_tags(0, I + 0, J + 0, K + 0)) << 0) |  //
                                  ((vertex_tags(0, I + 1, J + 0, K + 0)) << 1) |  //
                                  ((vertex_tags(0, I + 0, J + 1, K + 0)) << 2) |  //
                                  ((vertex_tags(0, I + 1, J + 1, K + 0)) << 3) |  //
                                  ((vertex_tags(0, I + 0, J + 0, K + 1)) << 4) |  //
                                  ((vertex_tags(0, I + 1, J + 0, K + 1)) << 5) |  //
                                  ((vertex_tags(0, I + 0, J + 1, K + 1)) << 6) |  //
                                  ((vertex_tags(0, I + 1, J + 1, K + 1)) << 7);
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

                    VOLUME_body->Insert(t7 + s);

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

    m_host_->GetRange("BODY_0")->append(VERTEX_body);
    m_host_->GetRange("BODY_1")->append(EDGE_body);
    m_host_->GetRange("BODY_2")->append(FACE_body);
    m_host_->GetRange("BODY_3")->append(VOLUME_body);

    m_host_->GetRange("BOUNDARY_0")->append(VERTEX_boundary);
    m_host_->GetRange("BOUNDARY_3")->append(VOLUME_boundary);

    m_host_->GetRange("PARA_BOUNDARY_1")->append(EDGE_PARA_boundary);
    m_host_->GetRange("PARA_BOUNDARY_2")->append(FACE_PARA_boundary);

    m_host_->GetRange("PERP_BOUNDARY_1")->append(EDGE_PERP_boundary);
    m_host_->GetRange("PERP_BOUNDARY_2")->append(FACE_PERP_boundary);
}
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_EBMESH_H
