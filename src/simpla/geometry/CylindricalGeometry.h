//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H


#include <vector>
#include <iomanip>
#include <simpla/SIMPLA_config.h>

#include <simpla/toolbox/macro.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/nTupleExt.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/type_cast.h>
#include <simpla/toolbox/Log.h>

#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>
#include "simpla/mesh/Domain.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */


struct CylindricalGeometry : public Domain
{

public:


    typedef Real scalar_type;
    static constexpr int ndims = 3;

    CylindricalGeometry() {}

    template<typename ...Args>
    CylindricalGeometry(Args &&...args):Domain(std::forward<Args>(args)...) {}

    ~CylindricalGeometry() {}


    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1> using data_block_type=
    mesh::DataBlockArray<TV, IFORM, DOF>;

private:
    mesh::AttributeView<Real, VERTEX, 3> m_vertics_{this, "vertices", "COORDINATES"};
    mesh::AttributeView<Real, VOLUME, 9> m_volume_{this, "volume", "NO_FILL"};
    mesh::AttributeView<Real, VOLUME, 9> m_dual_volume_{this, "dual_volume", "NO_FILL"};
    mesh::AttributeView<Real, VOLUME, 9> m_inv_volume_{this, "inv_volume", "NO_FILL"};
    mesh::AttributeView<Real, VOLUME, 9> m_inv_dual_volume_{this, "inv_dual_volume", "NO_FILL"};


public:


    point_type point(MeshEntityId id, point_type const &pr) const
    {
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


        auto const *d = static_cast<data_block_type<Real, VERTEX, 9> const *>(m_vertics_.data_block());

        auto i = MeshEntityIdCoder::unpack_index(id);
        Real r = pr[0], s = pr[1], t = pr[2];

        Real w0 = (1 - r) * (1 - s) * (1 - t);
        Real w1 = r * (1 - s) * (1 - t);
        Real w2 = (1 - r) * s * (1 - t);
        Real w3 = r * s * (1 - t);
        Real w4 = (1 - r) * (1 - s) * t;
        Real w5 = r * (1 - s) * t;
        Real w6 = (1 - r) * s * t;
        Real w7 = r * s * t;

        Real x = d->get(i[0]/**/, i[1]/**/, i[2]/**/, 0) * w0 +
                 d->get(i[0] + 1, i[1]/**/, i[2]/**/, 0) * w1 +
                 d->get(i[0]/**/, i[1] + 1, i[2]/**/, 0) * w2 +
                 d->get(i[0] + 1, i[1] + 1, i[2]/**/, 0) * w3 +
                 d->get(i[0]/**/, i[1]/**/, i[2] + 1, 0) * w4 +
                 d->get(i[0] + 1, i[1]/**/, i[2] + 1, 0) * w5 +
                 d->get(i[0]/**/, i[1] + 1, i[2] + 1, 0) * w6 +
                 d->get(i[0] + 1, i[1] + 1, i[2] + 1, 0) * w7;

        Real y = d->get(i[0]/**/, i[1]/**/, i[2]/**/, 1) * w0 +
                 d->get(i[0] + 1, i[1]/**/, i[2]/**/, 1) * w1 +
                 d->get(i[0]/**/, i[1] + 1, i[2]/**/, 1) * w2 +
                 d->get(i[0] + 1, i[1] + 1, i[2]/**/, 1) * w3 +
                 d->get(i[0]/**/, i[1]/**/, i[2] + 1, 1) * w4 +
                 d->get(i[0] + 1, i[1]/**/, i[2] + 1, 1) * w5 +
                 d->get(i[0]/**/, i[1] + 1, i[2] + 1, 1) * w6 +
                 d->get(i[0] + 1, i[1] + 1, i[2] + 1, 1) * w7;

        Real z = d->get(i[0]/**/, i[1]/**/, i[2]/**/, 2) * w0 +
                 d->get(i[0] + 1, i[1]/**/, i[2]/**/, 2) * w1 +
                 d->get(i[0]/**/, i[1] + 1, i[2]/**/, 2) * w2 +
                 d->get(i[0] + 1, i[1] + 1, i[2]/**/, 2) * w3 +
                 d->get(i[0]/**/, i[1]/**/, i[2] + 1, 2) * w4 +
                 d->get(i[0] + 1, i[1]/**/, i[2] + 1, 2) * w5 +
                 d->get(i[0]/**/, i[1] + 1, i[2] + 1, 2) * w6 +
                 d->get(i[0] + 1, i[1] + 1, i[2] + 1, 2) * w7;

        return point_type{x, y, z};
    }

    point_type point(MeshEntityId s) const
    {
        auto i = MeshEntityIdCoder::unpack_index(s);
        auto const *d = static_cast<data_block_type<Real, VERTEX, 9> const *>(m_vertics_.data_block());
        return point_type{d->get(i[0], i[1], i[2], 0),
                          d->get(i[0], i[1], i[2], 1),
                          d->get(i[0], i[1], i[2], 2)};
    }

    Real volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VOLUME, 9> const *>(m_volume_.data_block())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    Real dual_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VOLUME, 9> const *>(m_dual_volume_.data_block())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    Real inv_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VOLUME, 9> const *>(m_inv_volume_.data_block())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    Real inv_dual_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VOLUME, 9> const *>(m_inv_dual_volume_.data_block())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    void deploy() {}

    void initialize()
    {
        //        VERBOSE << mesh_block()->inv_dx() << mesh_block()->dx() << std::endl;

        m_vertics_.clear();
        m_volume_.clear();
        m_dual_volume_.clear();
        m_inv_volume_.clear();
        m_inv_dual_volume_.clear();
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
        auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_vertics_.data_block())->start();
        auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_vertics_.data_block())->count();

        index_type ib = m_start_[0];
        index_type ie = m_start_[0] + m_count_[0];
        index_type jb = m_start_[1];
        index_type je = m_start_[1] + m_count_[1];
        index_type kb = m_start_[2];
        index_type ke = m_start_[2] + m_count_[2];
        auto m_dx_ = m_mesh_block_->dx();
#define GET3(_NAME_, _I, _J, _K, _L)  ( static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(_NAME_.data_block()))->get(_I,_J,_K,_L)
#define GET9(_NAME_, _I, _J, _K, _L)  ( static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(_NAME_.data_block()))->get(_I,_J,_K,_L)

        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k)
                {
                    auto x = m_mesh_block_->point(i, j, k);

                    GET3(m_vertics_, i, j, k, 0) = (1 + x[0]) * std::cos(x[1]);
                    GET3(m_vertics_, i, j, k, 1) = (1 + x[0]) * std::sin(x[1]);
                    GET3(m_vertics_, i, j, k, 2) = x[2];
                    Real dr = m_dx_[0];
                    Real dl0 = m_dx_[1] * x[0];
                    Real dl1 = m_dx_[1] * (x[0] + m_dx_[0]);
                    Real dz = m_dx_[2];

                    GET9(m_volume_, i, j, k, 0) = 1.0;
                    GET9(m_volume_, i, j, k, 1) = dr;
                    GET9(m_volume_, i, j, k, 2) = dl0;
                    GET9(m_volume_, i, j, k, 3) = 0.5 * dr * (dl0 + dl1);
                    GET9(m_volume_, i, j, k, 4) = dz;
                    GET9(m_volume_, i, j, k, 5) = dr * dz;
                    GET9(m_volume_, i, j, k, 6) = dl0 * dz;
                    GET9(m_volume_, i, j, k, 7) = 0.5 * dr * (dl0 + dl1) * dz;
                    GET9(m_volume_, i, j, k, 8) = 1.0;

                    GET9(m_inv_volume_, i, j, k, 0) = 1.0 / GET9(m_volume_, i, j, k, 0);
                    GET9(m_inv_volume_, i, j, k, 1) = 1.0 / GET9(m_volume_, i, j, k, 1);
                    GET9(m_inv_volume_, i, j, k, 2) = 1.0 / GET9(m_volume_, i, j, k, 2);
                    GET9(m_inv_volume_, i, j, k, 3) = 1.0 / GET9(m_volume_, i, j, k, 3);
                    GET9(m_inv_volume_, i, j, k, 4) = 1.0 / GET9(m_volume_, i, j, k, 4);
                    GET9(m_inv_volume_, i, j, k, 5) = 1.0 / GET9(m_volume_, i, j, k, 5);
                    GET9(m_inv_volume_, i, j, k, 6) = 1.0 / GET9(m_volume_, i, j, k, 6);
                    GET9(m_inv_volume_, i, j, k, 7) = 1.0 / GET9(m_volume_, i, j, k, 7);
                    GET9(m_inv_volume_, i, j, k, 8) = 1.0 / GET9(m_volume_, i, j, k, 8);


                    dr = m_dx_[0];
                    dl0 = m_dx_[1] * (x[0] - 0.5 * m_dx_[0]);
                    dl1 = m_dx_[1] * (x[0] + 0.5 * m_dx_[0]);
                    dz = m_dx_[2];

                    GET9(m_dual_volume_, i, j, k, 7) = 1.0;
                    GET9(m_dual_volume_, i, j, k, 6) = dr;
                    GET9(m_dual_volume_, i, j, k, 5) = dl0;
                    GET9(m_dual_volume_, i, j, k, 4) = 0.5 * dr * (dl0 + dl1);
                    GET9(m_dual_volume_, i, j, k, 3) = dz;
                    GET9(m_dual_volume_, i, j, k, 2) = dr * dz;
                    GET9(m_dual_volume_, i, j, k, 1) = dl0 * dz;
                    GET9(m_dual_volume_, i, j, k, 0) = 0.5 * dr * (dl0 + dl1) * dz;
                    GET9(m_dual_volume_, i, j, k, 8) = 1.0;


                    GET9(m_inv_dual_volume_, i, j, k, 0) = 1.0 / GET9(m_dual_volume_, i, j, k, 0);
                    GET9(m_inv_dual_volume_, i, j, k, 1) = 1.0 / GET9(m_dual_volume_, i, j, k, 1);
                    GET9(m_inv_dual_volume_, i, j, k, 2) = 1.0 / GET9(m_dual_volume_, i, j, k, 2);
                    GET9(m_inv_dual_volume_, i, j, k, 3) = 1.0 / GET9(m_dual_volume_, i, j, k, 3);
                    GET9(m_inv_dual_volume_, i, j, k, 4) = 1.0 / GET9(m_dual_volume_, i, j, k, 4);
                    GET9(m_inv_dual_volume_, i, j, k, 5) = 1.0 / GET9(m_dual_volume_, i, j, k, 5);
                    GET9(m_inv_dual_volume_, i, j, k, 6) = 1.0 / GET9(m_dual_volume_, i, j, k, 6);
                    GET9(m_inv_dual_volume_, i, j, k, 7) = 1.0 / GET9(m_dual_volume_, i, j, k, 7);
                    GET9(m_inv_dual_volume_, i, j, k, 8) = 1.0 / GET9(m_dual_volume_, i, j, k, 8);


                }
#undef GET9
#undef GET3


    }


}; // struct  Mesh
}} // namespace simpla // namespace get_mesh


#endif //SIMPLA_CYLINDRICALRECTMESH_H
