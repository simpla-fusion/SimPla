//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H

#include <simpla/SIMPLA_config.h>
#include <iomanip>
#include <vector>

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>

#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/type_cast.h>
#include <simpla/utilities/type_traits.h>
namespace simpla {
namespace mesh {
struct CylindricalGeometry : public engine::Chart {
    SP_OBJECT_HEAD(CylindricalGeometry, engine::Chart)

    std::shared_ptr<data::DataTable> Serialize() const {
        auto p = engine::Chart::Serialize();
        p->SetValue<std::string>("Type", "CylindricalGeometry");
        return p;
    };
};
}
namespace engine {
using namespace simpla::data;
/**
 * @ingroup mesh
 * @brief Uniform structured get_mesh
 */
template <>
struct MeshView<mesh::CylindricalGeometry> : public engine::Mesh {
   public:
    SP_OBJECT_HEAD(MeshView<mesh::CylindricalGeometry>, engine::Mesh)
    typedef EntityId entity_id;

    static constexpr unsigned int NDIMS = 3;
    typedef Real scalar_type;
    MeshView(std::shared_ptr<mesh::CylindricalGeometry> c = nullptr) : engine::Mesh(c) {}
    MeshView(this_type const &other) = delete;
    virtual ~MeshView() {}

    //    this_type *Clone() const { return new this_type(*this); }
    virtual void Register(engine::AttributeGroup *other) { engine::AttributeGroup::Register(other); }

   private:
    Field<this_type, Real, VERTEX, 3> m_vertics_{this, "name"_ = "vertics", "COORDINATES"_};
    Field<this_type, Real, VOLUME, 9> m_volume_{this, "name"_ = "volume", "NO_FILL"_};
    Field<this_type, Real, VOLUME, 9> m_dual_volume_{this, "name"_ = "dual_volume", "NO_FILL"_};
    Field<this_type, Real, VOLUME, 9> m_inv_volume_{this, "name"_ = "inv_volume", "NO_FILL"_};
    Field<this_type, Real, VOLUME, 9> m_inv_dual_volume_{this, "name"_ = "inv_dual_volume", "NO_FILL"_};

   public:
    typedef EntityIdCoder M;

    virtual point_type point(index_type i, index_type j, index_type k) const {
        return point_type{m_vertics_[0](i, j, k), m_vertics_[1](i, j, k), m_vertics_[2](i, j, k)};
    };

    virtual point_type point(entity_id s) const { return point_type{}; /*m_mesh_block_->point(s); */ };

    virtual point_type point(entity_id id, point_type const &pr) const {
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

        auto i = EntityIdCoder::unpack_index(id);
        Real r = pr[0], s = pr[1], t = pr[2];

        Real w0 = (1 - r) * (1 - s) * (1 - t);
        Real w1 = r * (1 - s) * (1 - t);
        Real w2 = (1 - r) * s * (1 - t);
        Real w3 = r * s * (1 - t);
        Real w4 = (1 - r) * (1 - s) * t;
        Real w5 = r * (1 - s) * t;
        Real w6 = (1 - r) * s * t;
        Real w7 = r * s * t;

        Real x =
            m_vertics_(i[0] /**/, i[1] /**/, i[2] /**/, 0) * w0 + m_vertics_(i[0] + 1, i[1] /**/, i[2] /**/, 0) * w1 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] /**/, 0) * w2 + m_vertics_(i[0] + 1, i[1] + 1, i[2] /**/, 0) * w3 +
            m_vertics_(i[0] /**/, i[1] /**/, i[2] + 1, 0) * w4 + m_vertics_(i[0] + 1, i[1] /**/, i[2] + 1, 0) * w5 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] + 1, 0) * w6 + m_vertics_(i[0] + 1, i[1] + 1, i[2] + 1, 0) * w7;

        Real y =
            m_vertics_(i[0] /**/, i[1] /**/, i[2] /**/, 1) * w0 + m_vertics_(i[0] + 1, i[1] /**/, i[2] /**/, 1) * w1 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] /**/, 1) * w2 + m_vertics_(i[0] + 1, i[1] + 1, i[2] /**/, 1) * w3 +
            m_vertics_(i[0] /**/, i[1] /**/, i[2] + 1, 1) * w4 + m_vertics_(i[0] + 1, i[1] /**/, i[2] + 1, 1) * w5 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] + 1, 1) * w6 + m_vertics_(i[0] + 1, i[1] + 1, i[2] + 1, 1) * w7;

        Real z =
            m_vertics_(i[0] /**/, i[1] /**/, i[2] /**/, 2) * w0 + m_vertics_(i[0] + 1, i[1] /**/, i[2] /**/, 2) * w1 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] /**/, 2) * w2 + m_vertics_(i[0] + 1, i[1] + 1, i[2] /**/, 2) * w3 +
            m_vertics_(i[0] /**/, i[1] /**/, i[2] + 1, 2) * w4 + m_vertics_(i[0] + 1, i[1] /**/, i[2] + 1, 2) * w5 +
            m_vertics_(i[0] /**/, i[1] + 1, i[2] + 1, 2) * w6 + m_vertics_(i[0] + 1, i[1] + 1, i[2] + 1, 2) * w7;

        return point_type{x, y, z};
    }

    virtual Real volume(entity_id s) const { return m_volume_.at(s); }
    virtual Real dual_volume(entity_id s) const { return m_volume_.at(s); }
    virtual Real inv_volume(entity_id s) const { return m_volume_.at(s); }
    virtual Real inv_dual_volume(entity_id s) const { return m_volume_.at(s); }

    typedef EntityIdCoder m;
    virtual Range<entity_id> GetRange() const { return Range<entity_id>(); };

    template <typename TV>
    TV const &GetValue(std::shared_ptr<simpla::Array<TV, NDIMS>> const *a, entity_id const &s) const {
        return a[m::node_id(s)]->at(m::unpack_index(s));
    }
    template <typename TV>
    TV &GetValue(std::shared_ptr<simpla::Array<TV, NDIMS>> *a, entity_id const &s) const {
        return a[m::node_id(s)]->at(m::unpack_index(s));
    }

    void InitializeData(Real time_now = 0) {
        TIME_STAMP;

        m_vertics_.Clear();
        m_volume_.Clear();
        m_dual_volume_.Clear();
        m_inv_volume_.Clear();
        m_inv_dual_volume_.Clear();

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

};  // struct  Mesh
}  // namespace get_mesh
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICALRECTMESH_H
