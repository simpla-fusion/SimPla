//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALGEOMETRY_H
#define SIMPLA_CYLINDRICALGEOMETRY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/type_cast.h>
#include <simpla/utilities/type_traits.h>
#include <iomanip>
#include <vector>
#include "Mesh.h"
#include "SMesh.h"

namespace simpla {
namespace mesh {
struct CylindricalGeometry : public engine::Chart {
    SP_OBJECT_HEAD(CylindricalGeometry, engine::Chart)
    DECLARE_REGISTER_NAME("CylindricalGeometry");

    static constexpr int NDIMS = 3;
    std::shared_ptr<data::DataTable> Serialize() const override {
        auto p = engine::Chart::Serialize();
        p->SetValue<std::string>("Type", GetClassName());
        return p;
    };
};

using namespace simpla::data;

/**
 * @ingroup mesh
 * @brief Uniform structured get_mesh
 */
template <>
struct Mesh<CylindricalGeometry, SMesh> : public SMesh {
    typedef Mesh<CylindricalGeometry, SMesh> mesh_type;
    SP_OBJECT_HEAD(mesh_type, SMesh)

   public:
    using SMesh::GetChart;
    typedef Real scalar_type;

    explicit Mesh(std::shared_ptr<engine::Chart> c = nullptr)
        : SMesh(c != nullptr ? c : std::make_shared<CylindricalGeometry>()) {}
    ~Mesh() override = default;

    DECLARE_REGISTER_NAME("Mesh<CylindricalGeometry,SMesh>")

   public:
    using SMesh::point;

    void InitializeData(Real time_now) override {
        SMesh::InitializeData(time_now);

        auto &m_vertics_ = GetVertics();
        auto &m_volume_ = GetVolume();
        auto &m_dual_volume_ = GetDualVolume();
        auto &m_inv_volume_ = GetInvVolume();
        auto &m_inv_dual_volume_ = GetInvDualVolume();

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
        auto const *lower = &(std::get<0>(m_vertics_[0].GetIndexBox())[0]);
        auto const *upper = &(std::get<1>(m_vertics_[0].GetIndexBox())[0]);

        index_type ib = lower[0];
        index_type ie = upper[0];
        index_type jb = lower[1];
        index_type je = upper[1];
        index_type kb = lower[2];
        index_type ke = upper[2];
        point_type m_dx_ = GetChart()->GetDx();
        point_type x0 = GetChart()->GetOrigin();

        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) {
                    point_type x = GetChart()->inv_map(
                        point_type{static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k)});
                    m_vertics_[0](i, j, k) = x[0] * std::cos(x[1]);
                    m_vertics_[1](i, j, k) = x[0] * std::sin(x[1]);
                    m_vertics_[2](i, j, k) = x[2];
                }

        ib = std::get<0>(m_volume_[0].GetIndexBox())[0];
        jb = std::get<0>(m_volume_[0].GetIndexBox())[1];
        kb = std::get<0>(m_volume_[0].GetIndexBox())[2];
        ie = std::get<1>(m_volume_[0].GetIndexBox())[0];
        je = std::get<1>(m_volume_[0].GetIndexBox())[1];
        ke = std::get<1>(m_volume_[0].GetIndexBox())[2];

        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) {
                    point_type x = GetChart()->inv_map(
                        point_type{static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k)});

                    Real dr = m_dx_[0];
                    Real dl0 = m_dx_[1] * x[0];
                    Real dl1 = m_dx_[1] * (x[0] + m_dx_[0]);
                    Real dz = m_dx_[2];

                    m_volume_[0](i, j, k) = 1.0;
                    m_volume_[1](i, j, k) = dr;
                    m_volume_[2](i, j, k) = dl0;
                    m_volume_[3](i, j, k) = 0.5 * dr * (dl0 + dl1);
                    m_volume_[4](i, j, k) = dz;
                    m_volume_[5](i, j, k) = dr * dz;
                    m_volume_[6](i, j, k) = dl0 * dz;
                    m_volume_[7](i, j, k) = 0.5 * dr * (dl0 + dl1) * dz;
                    m_volume_[8](i, j, k) = 1.0;

                    m_inv_volume_[0](i, j, k) = 1.0 / m_volume_[0](i, j, k);
                    m_inv_volume_[1](i, j, k) = 1.0 / m_volume_[1](i, j, k);
                    m_inv_volume_[2](i, j, k) = 1.0 / m_volume_[2](i, j, k);
                    m_inv_volume_[3](i, j, k) = 1.0 / m_volume_[3](i, j, k);
                    m_inv_volume_[4](i, j, k) = 1.0 / m_volume_[4](i, j, k);
                    m_inv_volume_[5](i, j, k) = 1.0 / m_volume_[5](i, j, k);
                    m_inv_volume_[6](i, j, k) = 1.0 / m_volume_[6](i, j, k);
                    m_inv_volume_[7](i, j, k) = 1.0 / m_volume_[7](i, j, k);
                    m_inv_volume_[8](i, j, k) = 1.0 / m_volume_[8](i, j, k);

                    dr = m_dx_[0];
                    dl0 = m_dx_[1] * (x[0] - 0.5 * m_dx_[0]);
                    dl1 = m_dx_[1] * (x[0] + 0.5 * m_dx_[0]);
                    dz = m_dx_[2];

                    m_dual_volume_[7](i, j, k) = 1.0;
                    m_dual_volume_[6](i, j, k) = dr;
                    m_dual_volume_[5](i, j, k) = dl0;
                    m_dual_volume_[4](i, j, k) = 0.5 * dr * (dl0 + dl1);
                    m_dual_volume_[3](i, j, k) = dz;
                    m_dual_volume_[2](i, j, k) = dr * dz;
                    m_dual_volume_[1](i, j, k) = dl0 * dz;
                    m_dual_volume_[0](i, j, k) = 0.5 * dr * (dl0 + dl1) * dz;
                    m_dual_volume_[8](i, j, k) = 1.0;

                    m_inv_dual_volume_[0](i, j, k) = 1.0 / m_dual_volume_[0](i, j, k);
                    m_inv_dual_volume_[1](i, j, k) = 1.0 / m_dual_volume_[1](i, j, k);
                    m_inv_dual_volume_[2](i, j, k) = 1.0 / m_dual_volume_[2](i, j, k);
                    m_inv_dual_volume_[3](i, j, k) = 1.0 / m_dual_volume_[3](i, j, k);
                    m_inv_dual_volume_[4](i, j, k) = 1.0 / m_dual_volume_[4](i, j, k);
                    m_inv_dual_volume_[5](i, j, k) = 1.0 / m_dual_volume_[5](i, j, k);
                    m_inv_dual_volume_[6](i, j, k) = 1.0 / m_dual_volume_[6](i, j, k);
                    m_inv_dual_volume_[7](i, j, k) = 1.0 / m_dual_volume_[7](i, j, k);
                    m_inv_dual_volume_[8](i, j, k) = 1.0 / m_dual_volume_[8](i, j, k);
                }
    }
};  // struct  MeshBase

//    virtual point_type point(EntityId s) const override {
//        return GetChart()->inv_map(
//            point_type{static_cast<double>(s.x), static_cast<double>(s.y), static_cast<double>(s.z)});
//    };
//    virtual point_type point(EntityId id, point_type const &pr) const {

//
//        Real r = pr[0], s = pr[1], t = pr[2];
//
//        Real w0 = (1 - r) * (1 - s) * (1 - t);
//        Real w1 = r * (1 - s) * (1 - t);
//        Real w2 = (1 - r) * s * (1 - t);
//        Real w3 = r * s * (1 - t);
//        Real w4 = (1 - r) * (1 - s) * t;
//        Real w5 = r * (1 - s) * t;
//        Real w6 = (1 - r) * s * t;
//        Real w7 = r * s * t;
//
//        Real x =
//            m_vertics_(id.x /**/, id.y /**/, id.z /**/, 0) * w0 + m_vertics_(id.x + 1, id.y /**/, id.z /**/, 0) *
//            w1 +
//            m_vertics_(id.x /**/, id.y + 1, id.z /**/, 0) * w2 + m_vertics_(id.x + 1, id.y + 1, id.z /**/, 0) * w3
//            +
//            m_vertics_(id.x /**/, id.y /**/, id.z + 1, 0) * w4 + m_vertics_(id.x + 1, id.y /**/, id.z + 1, 0) * w5
//            +
//            m_vertics_(id.x /**/, id.y + 1, id.z + 1, 0) * w6 + m_vertics_(id.x + 1, id.y + 1, id.z + 1, 0) * w7;
//
//        Real y =
//            m_vertics_(id.x /**/, id.y /**/, id.z /**/, 1) * w0 + m_vertics_(id.x + 1, id.y /**/, id.z /**/, 1) *
//            w1 +
//            m_vertics_(id.x /**/, id.y + 1, id.z /**/, 1) * w2 + m_vertics_(id.x + 1, id.y + 1, id.z /**/, 1) * w3
//            +
//            m_vertics_(id.x /**/, id.y /**/, id.z + 1, 1) * w4 + m_vertics_(id.x + 1, id.y /**/, id.z + 1, 1) * w5
//            +
//            m_vertics_(id.x /**/, id.y + 1, id.z + 1, 1) * w6 + m_vertics_(id.x + 1, id.y + 1, id.z + 1, 1) * w7;
//
//        Real z =
//            m_vertics_(id.x /**/, id.y /**/, id.z /**/, 2) * w0 + m_vertics_(id.x + 1, id.y /**/, id.z /**/, 2) *
//            w1 +
//            m_vertics_(id.x /**/, id.y + 1, id.z /**/, 2) * w2 + m_vertics_(id.x + 1, id.y + 1, id.z /**/, 2) * w3
//            +
//            m_vertics_(id.x /**/, id.y /**/, id.z + 1, 2) * w4 + m_vertics_(id.x + 1, id.y /**/, id.z + 1, 2) * w5
//            +
//            m_vertics_(id.x /**/, id.y + 1, id.z + 1, 2) * w6 + m_vertics_(id.x + 1, id.y + 1, id.z + 1, 2) * w7;
//
//        return point_type{x, y, z};
//    }
}  // namespace mesh
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICALGEOMETRY_H
