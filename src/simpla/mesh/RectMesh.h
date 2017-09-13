//
// Created by salmon on 17-6-1.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H

#include "simpla/SIMPLA_config.h"

#include <simpla/algebra/Algebra.h>
#include <simpla/data/Data.h>
#include <simpla/engine/Attribute.h>
#include "StructuredMesh.h"

namespace simpla {
namespace mesh {

using namespace simpla::data;
/**
 * Axis are perpendicular
 */
template <typename THost>
struct RectMesh : public StructuredMesh {
    SP_DOMAIN_POLICY_HEAD(RectMesh);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);

    engine::AttributeT<Real, NODE> m_coordinates_{m_host_, "name"_ = "m_coordinates_", "COORDINATES"_};
    //     engine:: AttributeT< Real, NODE > m_vertices_{m_domain_, "name"_ = "m_vertices_","TEMP"_};

    engine::AttributeT<Real, NODE> m_vertex_volume_{m_host_, "name"_ = "m_vertex_volume_", "TEMP"_};
    engine::AttributeT<Real, NODE> m_vertex_inv_volume_{m_host_, "name"_ = "m_vertex_inv_volume_", "TEMP"_};
    engine::AttributeT<Real, NODE> m_vertex_dual_volume_{m_host_, "name"_ = "m_vertex_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, NODE> m_vertex_inv_dual_volume_{m_host_, "name"_ = "m_vertex_inv_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, CELL> m_volume_volume_{m_host_, "name"_ = "m_volume_volume_", "TEMP"_};
    engine::AttributeT<Real, CELL> m_volume_inv_volume_{m_host_, "name"_ = "m_volume_inv_volume_", "TEMP"_};
    engine::AttributeT<Real, CELL> m_volume_dual_volume_{m_host_, "name"_ = "m_volume_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, CELL> m_volume_inv_dual_volume_{m_host_, "name"_ = "m_volume_inv_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, EDGE> m_edge_volume_{m_host_, "name"_ = "m_edge_volume_", "TEMP"_};
    engine::AttributeT<Real, EDGE> m_edge_inv_volume_{m_host_, "name"_ = "m_edge_inv_volume_", "TEMP"_};
    engine::AttributeT<Real, EDGE> m_edge_dual_volume_{m_host_, "name"_ = "m_edge_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, EDGE> m_edge_inv_dual_volume_{m_host_, "name"_ = "m_edge_inv_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, FACE> m_face_volume_{m_host_, "name"_ = "m_face_volume_", "TEMP"_};
    engine::AttributeT<Real, FACE> m_face_inv_volume_{m_host_, "name"_ = "m_face_inv_volume_", "TEMP"_};
    engine::AttributeT<Real, FACE> m_face_dual_volume_{m_host_, "name"_ = "m_face_dual_volume_", "TEMP"_};
    engine::AttributeT<Real, FACE> m_face_inv_dual_volume_{m_host_, "name"_ = "m_face_inv_dual_volume_", "TEMP"_};
};
template <typename THost>
RectMesh<THost>::RectMesh(THost* h) : m_host_(h) {
    h->PreInitialCondition.Connect([=](engine::DomainBase* self, Real time_now) {
        if (auto* p = dynamic_cast<RectMesh<THost>*>(self)) { p->InitialCondition(time_now); }
    });
}
template <typename THost>
RectMesh<THost>::~RectMesh() {}

template <typename THost>
std::shared_ptr<data::DataNode> RectMesh<THost>::Serialize() const {
    return nullptr;
}
template <typename THost>
void RectMesh<THost>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {}
template <typename THost>
void RectMesh<THost>::InitialCondition(Real time_now) {
    auto chart = this->GetChart();

    //    m_coordinates_ = [&](index_type x, index_type y, index_type z) { return chart->global_coordinates(x, y, z,
    //    0b0); };
    //    //    m_vertices_ = [&](point_type const &x) { return (x); };
    //    m_vertex_volume_.Initialize();
    //    m_vertex_inv_volume_.Initialize();
    //    m_vertex_dual_volume_.Initialize();
    //    m_vertex_inv_dual_volume_.Initialize();
    //
    //    m_volume_volume_.Initialize();
    //    m_volume_inv_volume_.Initialize();
    //    m_volume_dual_volume_.Initialize();
    //    m_volume_inv_dual_volume_.Initialize();
    //
    //    m_edge_volume_.Initialize();
    //    m_edge_inv_volume_.Initialize();
    //    m_edge_dual_volume_.Initialize();
    //    m_edge_inv_dual_volume_.Initialize();
    //
    //    m_face_volume_.Initialize();
    //    m_face_inv_volume_.Initialize();
    //    m_face_dual_volume_.Initialize();
    //    m_face_inv_dual_volume_.Initialize();
    //
    //    /**
    //     *\verbatim
    //     *                ^y (dl)
    //     *               /
    //     *   (dz) z     /
    //     *        ^    /
    //     *        |  110-------------111
    //     *        |  /|              /|
    //     *        | / |             / |
    //     *        |/  |            /  |
    //     *       100--|----------101  |
    //     *        | m |           |   |
    //     *        |  010----------|--011
    //     *        |  /            |  /
    //     *        | /             | /
    //     *        |/              |/
    //     *       000-------------001---> x (dr)
    //     *
    //     *\endverbatim
    //     */
    //
    //    m_vertex_volume_ = 1.0;
    //    m_vertex_inv_volume_ = 1.0;
    //    m_vertex_dual_volume_ = [&](index_type x, index_type y, index_type z, int tag) -> Real {
    //        return chart->volume(chart->local_coordinates(x - 1, y - 1, z - 1, 0b111),
    //                             chart->local_coordinates(x, y, z, 0b111));
    //    };
    //    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;
    //
    //    m_volume_volume_ = [&](index_type x, index_type y, index_type z, int tag) -> Real {
    //        return chart->volume(chart->local_coordinates(x, y, z, 0b0),
    //                             chart->local_coordinates(x + 1, y + 1, z + 1, 0b0));
    //    };
    //    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    //    m_volume_dual_volume_ = 1.0;
    //    m_volume_inv_dual_volume_ = 1.0;
    //
    //    m_edge_volume_ = [&](index_type x, index_type y, index_type z, int w) -> Real {
    //        return chart->length(
    //            chart->local_coordinates(x, y, z, 0b0),
    //            chart->local_coordinates(x + (w == 0b001 ? 1 : 0), y + (w == 0b010 ? 1 : 0), z + (w == 0b100 ? 1 : 0),
    //            0b0),
    //            EntityIdCoder::m_id_to_sub_index_[w]);
    //    };
    //    m_edge_inv_volume_ = 1.0 / m_edge_volume_;
    //
    //    m_edge_dual_volume_ = [&](index_type x, index_type y, index_type z, int w) -> Real {
    //        return chart->area(chart->local_coordinates(x - (w != 0b001 ? 1 : 0), y - (w != 0b010 ? 1 : 0),
    //                                                    z - (w != 0b100 ? 1 : 0), 0b111),
    //                           chart->local_coordinates(x, y, z, 0b111), EntityIdCoder::m_id_to_sub_index_[w]);
    //    };
    //    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;
    //
    //    m_face_volume_ = [&](index_type x, index_type y, index_type z, int w) -> Real {
    //        return chart->area(
    //            chart->local_coordinates(x, y, z, 0b0),
    //            chart->local_coordinates(x + (w != 0b110 ? 1 : 0), y + (w != 0b101 ? 1 : 0), z + (w != 0b011 ? 1 : 0),
    //            0b0),
    //            EntityIdCoder::m_id_to_sub_index_[w]);
    //
    //    };
    //
    //    m_face_inv_volume_ = 1.0 / m_face_volume_;
    //
    //    m_face_dual_volume_ = [&](index_type x, index_type y, index_type z, int w) -> Real {
    //        return chart->length(chart->local_coordinates(x - (w == 0b110 ? 1 : 0), y - (w == 0b101 ? 1 : 0),
    //                                                      z - (w == 0b011 ? 1 : 0), 0b111),
    //                             chart->local_coordinates(x, y, z, 0b111), EntityIdCoder::m_id_to_sub_index_[w]);
    //    };
    //    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

template <typename THost>
void RectMesh<THost>::BoundaryCondition(Real time_now, Real time_dt) {
    //    this->FillRange(m_vertex_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_vertex_dual_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_vertex_inv_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_vertex_inv_dual_volume_, 0, "PATCH_BOUNDARY_");
    //
    //    this->FillRange(m_edge_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_edge_dual_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_edge_inv_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_edge_inv_dual_volume_, 0, "PATCH_BOUNDARY_");
    //
    //    this->FillRange(m_face_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_face_dual_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_face_inv_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_face_inv_dual_volume_, 0, "PATCH_BOUNDARY_");
    //
    //    this->FillRange(m_volume_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_volume_dual_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_volume_inv_volume_, 0, "PATCH_BOUNDARY_");
    //    this->FillRange(m_volume_inv_dual_volume_, 0, "PATCH_BOUNDARY_");
}

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_RECTMESH_H
