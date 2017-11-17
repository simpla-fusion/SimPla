//
// Created by salmon on 17-6-1.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H

#include "simpla/SIMPLA_config.h"

#include <simpla/algebra/Algebra.h>
#include <simpla/data/Data.h>
#include <simpla/engine/Attribute.h>
#include <simpla/parallel/MPIComm.h>
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

    engine::AttributeT<Real, NODE, 3> m_coordinates_{m_host_, "Name"_ = "_COORDINATES_", "COORDINATES"_, "LOCAL"_,
                                                     "CheckPoint"_};
    //     engine:: AttributeT< Real, NODE > m_vertices_{m_domain_, "Name"_ = "m_vertices_","LOCAL"_};

    engine::AttributeT<Real, NODE> m_node_volume_{m_host_, "Name"_ = "m_node_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, NODE> m_node_inv_volume_{m_host_, "Name"_ = "m_node_inv_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, NODE> m_node_dual_volume_{m_host_, "Name"_ = "m_node_dual_volume_", "LOCAL"_,
                                                       "CheckPoint"_};
    engine::AttributeT<Real, NODE> m_node_inv_dual_volume_{m_host_, "Name"_ = "m_node_inv_dual_volume_", "LOCAL"_,
                                                           "CheckPoint"_};
    engine::AttributeT<Real, CELL> m_cell_volume_{m_host_, "Name"_ = "m_cell_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, CELL> m_cell_inv_volume_{m_host_, "Name"_ = "m_cell_inv_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, CELL> m_cell_dual_volume_{m_host_, "Name"_ = "m_cell_dual_volume_", "LOCAL"_,
                                                       "CheckPoint"_};
    engine::AttributeT<Real, CELL> m_cell_inv_dual_volume_{m_host_, "Name"_ = "m_cell_inv_dual_volume_", "LOCAL"_,
                                                           "CheckPoint"_};
    engine::AttributeT<Real, EDGE> m_edge_volume_{m_host_, "Name"_ = "m_edge_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, EDGE> m_edge_inv_volume_{m_host_, "Name"_ = "m_edge_inv_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, EDGE> m_edge_dual_volume_{m_host_, "Name"_ = "m_edge_dual_volume_", "LOCAL"_,
                                                       "CheckPoint"_};
    engine::AttributeT<Real, EDGE> m_edge_inv_dual_volume_{m_host_, "Name"_ = "m_edge_inv_dual_volume_", "LOCAL"_,
                                                           "CheckPoint"_};
    engine::AttributeT<Real, FACE> m_face_volume_{m_host_, "Name"_ = "m_face_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, FACE> m_face_inv_volume_{m_host_, "Name"_ = "m_face_inv_volume_", "LOCAL"_, "CheckPoint"_};
    engine::AttributeT<Real, FACE> m_face_dual_volume_{m_host_, "Name"_ = "m_face_dual_volume_", "LOCAL"_,
                                                       "CheckPoint"_};
    engine::AttributeT<Real, FACE> m_face_inv_dual_volume_{m_host_, "Name"_ = "m_face_inv_dual_volume_", "LOCAL"_,
                                                           "CheckPoint"_};
};
template <typename THost>
RectMesh<THost>::RectMesh(THost* h) : m_host_(h) {
    h->PreInitialCondition.Connect([=](engine::DomainBase* self, Real time_now) {
        if (auto* p = dynamic_cast<RectMesh<THost>*>(self)) { p->InitialCondition(time_now); }
    });
    h->OnSerialize.Connect([=](engine::DomainBase const* self, std::shared_ptr<simpla::data::DataEntry>& tdb) {
        if (auto const* p = dynamic_cast<RectMesh<THost> const*>(self)) { tdb->Set(p->Serialize()); }
    });
}
template <typename THost>
RectMesh<THost>::~RectMesh() {}

template <typename THost>
std::shared_ptr<data::DataEntry> RectMesh<THost>::Serialize() const {
    auto res = data::DataEntry::New(data::DataEntry::DN_TABLE);
    res->SetValue("Topology", "3DSMesh");
    return res;
}
template <typename THost>
void RectMesh<THost>::Deserialize(std::shared_ptr<data::DataEntry> const& cfg) {}
template <typename THost>
void RectMesh<THost>::InitialCondition(Real time_now) {
    auto chart = this->GetChart();
    m_host_->InitializeAttribute(&m_coordinates_);
    //    m_vertices_ = [&](point_type const& x) { return (x); };
    m_host_->InitializeAttribute(&m_node_volume_);
    m_host_->InitializeAttribute(&m_node_inv_volume_);
    m_host_->InitializeAttribute(&m_node_dual_volume_);
    m_host_->InitializeAttribute(&m_node_inv_dual_volume_);

    m_host_->InitializeAttribute(&m_cell_volume_);
    m_host_->InitializeAttribute(&m_cell_inv_volume_);
    m_host_->InitializeAttribute(&m_cell_dual_volume_);
    m_host_->InitializeAttribute(&m_cell_inv_dual_volume_);

    m_host_->InitializeAttribute(&m_edge_volume_);
    m_host_->InitializeAttribute(&m_edge_inv_volume_);
    m_host_->InitializeAttribute(&m_edge_dual_volume_);
    m_host_->InitializeAttribute(&m_edge_inv_dual_volume_);

    m_host_->InitializeAttribute(&m_face_volume_);
    m_host_->InitializeAttribute(&m_face_inv_volume_);
    m_host_->InitializeAttribute(&m_face_dual_volume_);
    m_host_->InitializeAttribute(&m_face_inv_dual_volume_);

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

    m_coordinates_ = [&](index_type x, index_type y, index_type z) { return chart->global_coordinates(0b0, x, y, z); };

    m_node_volume_ = 1.0;
    m_node_inv_volume_ = 1.0;
    m_node_dual_volume_ = [&](index_type x, index_type y, index_type z) -> Real {
        return chart->volume(chart->local_coordinates(0b111, x - 1, y - 1, z - 1),
                             chart->local_coordinates(0b111, x, y, z));
    };
    m_node_inv_dual_volume_ = 1.0 / m_node_dual_volume_;

    m_cell_volume_ = [&](index_type x, index_type y, index_type z) -> Real {
        return chart->volume(chart->local_coordinates(0b0, x, y, z),
                             chart->local_coordinates(0b0, x + 1, y + 1, z + 1));
    };
    m_cell_inv_volume_ = 1.0 / m_cell_volume_;
    m_cell_dual_volume_ = 1.0;
    m_cell_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](int w, index_type x, index_type y, index_type z) -> Real {
        return chart->length(
            chart->local_coordinates(0b0, x, y, z),
            chart->local_coordinates(0b0, x + (w == 0 ? 1 : 0), y + (w == 1 ? 1 : 0), z + (w == 2 ? 1 : 0)), w);
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](int w, index_type x, index_type y, index_type z) -> Real {
        return chart->area(
            chart->local_coordinates(0b111, x - (w != 0 ? 1 : 0), y - (w != 1 ? 1 : 0), z - (w != 2 ? 1 : 0)),
            chart->local_coordinates(0b111, x, y, z), w);
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](int w, index_type x, index_type y, index_type z) -> Real {
        return chart->area(
            chart->local_coordinates(0b0, x, y, z),
            chart->local_coordinates(0b0, x + (w != 0 ? 1 : 0), y + (w != 1 ? 1 : 0), z + (w != 2 ? 1 : 0)), w);

    };
    m_face_inv_volume_ = 1.0 / m_face_volume_;

    m_face_dual_volume_ = [&](int w, index_type x, index_type y, index_type z) -> Real {
        return chart->length(
            chart->local_coordinates(0b111, x - (w == 0 ? 1 : 0), y - (w == 1 ? 1 : 0), z - (w == 2 ? 1 : 0)),
            chart->local_coordinates(0b111, x, y, z), w);
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

template <typename THost>
void RectMesh<THost>::BoundaryCondition(Real time_now, Real time_dt) {
    m_host_->FillRange(m_node_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_node_dual_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_node_inv_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_node_inv_dual_volume_, 0, "PATCH_BOUNDARY_");

    m_host_->FillRange(m_edge_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_edge_dual_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_edge_inv_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_edge_inv_dual_volume_, 0, "PATCH_BOUNDARY_");

    m_host_->FillRange(m_face_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_face_dual_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_face_inv_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_face_inv_dual_volume_, 0, "PATCH_BOUNDARY_");

    m_host_->FillRange(m_cell_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_cell_dual_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_cell_inv_volume_, 0, "PATCH_BOUNDARY_");
    m_host_->FillRange(m_cell_inv_dual_volume_, 0, "PATCH_BOUNDARY_");
}

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_RECTMESH_H
