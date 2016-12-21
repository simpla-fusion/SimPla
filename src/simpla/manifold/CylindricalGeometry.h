//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H


#include <vector>
#include <iomanip>
#include <simpla/SIMPLA_config.h>

#include <simpla/toolbox/macro.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/type_cast.h>
#include <simpla/toolbox/Log.h>

#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>
#include "simpla/mesh/Chart.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */


struct CylindricalGeometry : public Chart
{

public:
    SP_OBJECT_HEAD(CylindricalGeometry, Chart)

    static constexpr bool is_frame_bundle = true;

    typedef Real scalar_type;

    static constexpr int ndims = 3;

    CylindricalGeometry() {}

    virtual ~CylindricalGeometry() {}


    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1> using data_block_type=mesh::DataBlockArray<TV, IFORM, DOF>;

//private:
    DataAttribute<Real, VERTEX, 3> m_vertics_{this, "name=vertics;COORDINATES"};
    DataAttribute<Real, VOLUME, 9> m_volume_{this, "name=volume;NO_FILL"};
    DataAttribute<Real, VOLUME, 9> m_dual_volume_{this, "name=dual_volume;NO_FILL"};
    DataAttribute<Real, VOLUME, 9> m_inv_volume_{this, "name=inv_volume;NO_FILL"};
    DataAttribute<Real, VOLUME, 9> m_inv_dual_volume_{this, "name=inv_dual_volume;NO_FILL"};

public:
    typedef mesh::MeshEntityIdCoder M;

    virtual point_type point(index_type i, index_type j, index_type k) const
    {
        return point_type{
                m_vertics_.get(i, j, k, 0),
                m_vertics_.get(i, j, k, 1),
                m_vertics_.get(i, j, k, 2)
        };
    };

    virtual point_type point(MeshEntityId s) const { return m_mesh_block_->point(s); };

    virtual point_type point(MeshEntityId id, point_type const &pr) const
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

        Real x = m_vertics_.get(i[0]/**/, i[1]/**/, i[2]/**/, 0) * w0 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2]/**/, 0) * w1 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2]/**/, 0) * w2 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2]/**/, 0) * w3 +
                 m_vertics_.get(i[0]/**/, i[1]/**/, i[2] + 1, 0) * w4 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2] + 1, 0) * w5 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2] + 1, 0) * w6 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2] + 1, 0) * w7;

        Real y = m_vertics_.get(i[0]/**/, i[1]/**/, i[2]/**/, 1) * w0 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2]/**/, 1) * w1 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2]/**/, 1) * w2 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2]/**/, 1) * w3 +
                 m_vertics_.get(i[0]/**/, i[1]/**/, i[2] + 1, 1) * w4 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2] + 1, 1) * w5 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2] + 1, 1) * w6 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2] + 1, 1) * w7;

        Real z = m_vertics_.get(i[0]/**/, i[1]/**/, i[2]/**/, 2) * w0 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2]/**/, 2) * w1 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2]/**/, 2) * w2 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2]/**/, 2) * w3 +
                 m_vertics_.get(i[0]/**/, i[1]/**/, i[2] + 1, 2) * w4 +
                 m_vertics_.get(i[0] + 1, i[1]/**/, i[2] + 1, 2) * w5 +
                 m_vertics_.get(i[0]/**/, i[1] + 1, i[2] + 1, 2) * w6 +
                 m_vertics_.get(i[0] + 1, i[1] + 1, i[2] + 1, 2) * w7;

        return point_type{x, y, z};
    }


    virtual Real volume(MeshEntityId s) const { return m_volume_.get(M::sw(s, M::node_id(s))); }

    virtual Real dual_volume(MeshEntityId s) const { return m_dual_volume_.get(M::sw(s, M::node_id(s))); }

    virtual Real inv_volume(MeshEntityId s) const { return m_inv_volume_.get(M::sw(s, M::node_id(s))); }

    virtual Real inv_dual_volume(MeshEntityId s) const { return m_inv_dual_volume_.get(M::sw(s, M::node_id(s))); }


    virtual void initialize(Real data_time, Real dt)
    {
        base_type::initialize(data_time, 0);

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
        auto m_start_ = std::dynamic_pointer_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> >(
                m_vertics_.data())->start();
        auto m_count_ = std::dynamic_pointer_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> >(
                m_vertics_.data())->count();

        index_type ib = m_start_[0];
        index_type ie = m_start_[0] + m_count_[0];
        index_type jb = m_start_[1];
        index_type je = m_start_[1] + m_count_[1];
        index_type kb = m_start_[2];
        index_type ke = m_start_[2] + m_count_[2];
        auto m_dx_ = mesh_block()->dx();


        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k)
                {
                    auto x = mesh_block()->point(i, j, k);
                    m_vertics_.get(i, j, k, 0) = x[0] * std::cos(x[1]);
                    m_vertics_.get(i, j, k, 1) = x[0] * std::sin(x[1]);
                    m_vertics_.get(i, j, k, 2) = x[2];
                    Real dr = m_dx_[0];
                    Real dl0 = m_dx_[1] * x[0];
                    Real dl1 = m_dx_[1] * (x[0] + m_dx_[0]);
                    Real dz = m_dx_[2];

                    m_volume_.get(i, j, k, 0) = 1.0;
                    m_volume_.get(i, j, k, 1) = dr;
                    m_volume_.get(i, j, k, 2) = dl0;
                    m_volume_.get(i, j, k, 3) = 0.5 * dr * (dl0 + dl1);
                    m_volume_.get(i, j, k, 4) = dz;
                    m_volume_.get(i, j, k, 5) = dr * dz;
                    m_volume_.get(i, j, k, 6) = dl0 * dz;
                    m_volume_.get(i, j, k, 7) = 0.5 * dr * (dl0 + dl1) * dz;
                    m_volume_.get(i, j, k, 8) = 1.0;

                    m_inv_volume_.get(i, j, k, 0) = 1.0 / m_volume_.get(i, j, k, 0);
                    m_inv_volume_.get(i, j, k, 1) = 1.0 / m_volume_.get(i, j, k, 1);
                    m_inv_volume_.get(i, j, k, 2) = 1.0 / m_volume_.get(i, j, k, 2);
                    m_inv_volume_.get(i, j, k, 3) = 1.0 / m_volume_.get(i, j, k, 3);
                    m_inv_volume_.get(i, j, k, 4) = 1.0 / m_volume_.get(i, j, k, 4);
                    m_inv_volume_.get(i, j, k, 5) = 1.0 / m_volume_.get(i, j, k, 5);
                    m_inv_volume_.get(i, j, k, 6) = 1.0 / m_volume_.get(i, j, k, 6);
                    m_inv_volume_.get(i, j, k, 7) = 1.0 / m_volume_.get(i, j, k, 7);
                    m_inv_volume_.get(i, j, k, 8) = 1.0 / m_volume_.get(i, j, k, 8);


                    dr = m_dx_[0];
                    dl0 = m_dx_[1] * (x[0] - 0.5 * m_dx_[0]);
                    dl1 = m_dx_[1] * (x[0] + 0.5 * m_dx_[0]);
                    dz = m_dx_[2];

                    m_dual_volume_.get(i, j, k, 7) = 1.0;
                    m_dual_volume_.get(i, j, k, 6) = dr;
                    m_dual_volume_.get(i, j, k, 5) = dl0;
                    m_dual_volume_.get(i, j, k, 4) = 0.5 * dr * (dl0 + dl1);
                    m_dual_volume_.get(i, j, k, 3) = dz;
                    m_dual_volume_.get(i, j, k, 2) = dr * dz;
                    m_dual_volume_.get(i, j, k, 1) = dl0 * dz;
                    m_dual_volume_.get(i, j, k, 0) = 0.5 * dr * (dl0 + dl1) * dz;
                    m_dual_volume_.get(i, j, k, 8) = 1.0;


                    m_inv_dual_volume_.get(i, j, k, 0) = 1.0 / m_dual_volume_.get(i, j, k, 0);
                    m_inv_dual_volume_.get(i, j, k, 1) = 1.0 / m_dual_volume_.get(i, j, k, 1);
                    m_inv_dual_volume_.get(i, j, k, 2) = 1.0 / m_dual_volume_.get(i, j, k, 2);
                    m_inv_dual_volume_.get(i, j, k, 3) = 1.0 / m_dual_volume_.get(i, j, k, 3);
                    m_inv_dual_volume_.get(i, j, k, 4) = 1.0 / m_dual_volume_.get(i, j, k, 4);
                    m_inv_dual_volume_.get(i, j, k, 5) = 1.0 / m_dual_volume_.get(i, j, k, 5);
                    m_inv_dual_volume_.get(i, j, k, 6) = 1.0 / m_dual_volume_.get(i, j, k, 6);
                    m_inv_dual_volume_.get(i, j, k, 7) = 1.0 / m_dual_volume_.get(i, j, k, 7);
                    m_inv_dual_volume_.get(i, j, k, 8) = 1.0 / m_dual_volume_.get(i, j, k, 8);


                }
    }

}; // struct  Mesh
}} // namespace simpla // namespace get_mesh


#endif //SIMPLA_CYLINDRICALRECTMESH_H
