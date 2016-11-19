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
#include "Geometry.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */


struct CylindricalGeometry : public GeometryBase
{
private:
    typedef CylindricalGeometry this_type;
    typedef MeshBlock base_type;

public:
    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return std::string("CylindricalGeometry"); }


public:
    static constexpr int ndims = 3;

    template<typename ...Args>
    explicit CylindricalGeometry(Args &&...args):GeometryBase(std::forward<Args>(args)...) {}

    ~CylindricalGeometry() {}


    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1> using data_block_type=
    mesh::DataBlockArray<TV, IFORM, DOF>;

    mesh::AttributeView<Real, VERTEX, 3> m_vertics_;
    mesh::AttributeView<Real, VERTEX, 9> m_volume_;
    mesh::AttributeView<Real, VERTEX, 9> m_dual_volume_;
    mesh::AttributeView<Real, VERTEX, 9> m_inv_volume_;
    mesh::AttributeView<Real, VERTEX, 9> m_inv_dual_volume_;


public:
    void connect(AttributeHolder *holder)
    {
        GeometryBase::connect(holder);
        m_vertics_.connect(holder, "vertices", "COORDINATES");
        m_volume_.connect(holder, "volume", "NO_FILL");
        m_dual_volume_.connect(holder, "dual_volume", "NO_FILL");
        m_inv_volume_.connect(holder, "inv_volume", "NO_FILL");
        m_inv_dual_volume_.connect(holder, "inv_dual_volume", "NO_FILL");
    }

    template<typename ...Args>
    point_type point(Args &&...args) const { return m_mesh_->point(std::forward<Args>(args)...); }

    virtual Real volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VERTEX, 9> const *>(m_volume_.data())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    virtual Real
    dual_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VERTEX, 9> const *>(m_dual_volume_.data())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    virtual Real
    inv_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VERTEX, 9> const *>(m_inv_volume_.data())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    virtual Real
    inv_dual_volume(MeshEntityId s) const
    {
        return static_cast<data_block_type<Real, VERTEX, 9> const *>(m_inv_dual_volume_.data())->
                get(MeshEntityIdCoder::unpack_index4_nodeid(s));
    }

    void deploy() {}

    void initialize()
    {
        //        VERBOSE << mesh_block()->inv_dx() << mesh_block()->dx() << std::endl;
        ASSERT(m_mesh_ != nullptr);

        m_vertics_.clear();
        m_volume_.clear();
        m_dual_volume_.clear();
        m_inv_volume_.clear();
        m_inv_dual_volume_.clear();

        /**
             *\verbatim
             *                ^y
             *               /
             *        z     /
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
             *       000-------------001---> x
             *
             *\endverbatim
             */

        auto *d = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>( m_vertics_.data());

        CHECK(d->ndims());
        VERBOSE << d->count()[0] << " , " << d->count()[1] << " , " << d->count()[2] << " , " << d->count()[3] << " , "
                << std::endl;
        d->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    auto x = m_mesh_->point(i, j, k);

                    double res = 0.0;
                    switch (l)
                    {
                        case 0:
                            res = (1 + x[0]) * std::cos(x[1]);
                            break;
                        case 1:
                            res = (1 + x[0]) * std::sin(x[1]);
                            break;
                        case 2:
                            res = x[2];
                            break;
                        default :
                            break;
                    }
                    return res;

                });


        for (int i = 0; i < ndims; ++i)
        {



//            m_dx_[i] = (m_dims_[i] <= 1) ? 1 : (m_coords_upper_[i] - m_coords_lower_[i]) /
//                                               static_cast<Real>( m_dims_[i]);
//
//            m_inv_dx_[i] = (m_dims_[i] <= 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];
//
//            m_l2g_scale_[i] = (m_dims_[i] <= 1) ? 0 : m_dx_[i];
//            m_l2g_shift_ = m_coords_lower_;
//
//            m_g2l_scale_[i] = (m_dims_[i] <= 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];
//            m_g2l_shift_[i] = (m_dims_[i] <= 1) ? 0 : -m_coords_lower_[i] * m_g2l_scale_[i];

        }


//        m_volume_[0 /*000*/] = 1;
//        m_volume_[1 /*001*/] = m_dx_[0];
//        m_volume_[2 /*010*/] = m_dx_[1];
//        m_volume_[4 /*100*/] = m_dx_[2];
//        m_volume_[3 /*011*/] = m_dx_[0] * m_dx_[1];
//        m_volume_[5 /*101*/] = m_dx_[2] * m_dx_[0];
//        m_volume_[6 /*110*/] = m_dx_[1] * m_dx_[2];
//        m_volume_[7 /*110*/] = m_dx_[0] * m_dx_[1] * m_dx_[2];
//
//
//        m_dual_volume_[0 /*000*/] = m_volume_[7];
//        m_dual_volume_[1 /*001*/] = m_volume_[6];
//        m_dual_volume_[2 /*010*/] = m_volume_[5];
//        m_dual_volume_[4 /*100*/] = m_volume_[3];
//        m_dual_volume_[3 /*011*/] = m_volume_[4];
//        m_dual_volume_[5 /*101*/] = m_volume_[2];
//        m_dual_volume_[6 /*110*/] = m_volume_[1];
//        m_dual_volume_[7 /*110*/] = m_volume_[0];
//
//
//        m_inv_dx_[0] = (m_dims_[0] == 1) ? 1 : 1 / m_dx_[0];
//        m_inv_dx_[1] = (m_dims_[1] == 1) ? 1 : 1 / m_dx_[1];
//        m_inv_dx_[2] = (m_dims_[2] == 1) ? 1 : 1 / m_dx_[2];
//
//
//        m_inv_volume_[0 /*000*/] = 1;
//        m_inv_volume_[1 /*001*/] = m_inv_dx_[0];
//        m_inv_volume_[2 /*010*/] = m_inv_dx_[1];
//        m_inv_volume_[4 /*100*/] = m_inv_dx_[2];
//        m_inv_volume_[3 /*011*/] = m_inv_dx_[0] * m_inv_dx_[1];
//        m_inv_volume_[5 /*101*/] = m_inv_dx_[2] * m_inv_dx_[0];
//        m_inv_volume_[6 /*110*/] = m_inv_dx_[1] * m_inv_dx_[2];
//        m_inv_volume_[7 /*110*/] = m_inv_dx_[0] * m_inv_dx_[1] * m_inv_dx_[2];
//
//
//        m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
//        m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
//        m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
//        m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
//        m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
//        m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
//        m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
//        m_inv_dual_volume_[7 /*110*/] = m_inv_volume_[0];
//
//
//        m_inv_dx_[0] = (m_dims_[0] <= 1) ? 0 : m_inv_dx_[0];
//        m_inv_dx_[1] = (m_dims_[1] <= 1) ? 0 : m_inv_dx_[1];
//        m_inv_dx_[2] = (m_dims_[2] <= 1) ? 0 : m_inv_dx_[2];
//
//
//        m_inv_volume_[1 /*001*/] = (m_dims_[0] <= 1) ? 0 : m_inv_dx_[0];
//        m_inv_volume_[2 /*010*/] = (m_dims_[1] <= 1) ? 0 : m_inv_dx_[1];
//        m_inv_volume_[4 /*100*/] = (m_dims_[2] <= 1) ? 0 : m_inv_dx_[2];
//
//

    }


}; // struct  Mesh
}} // namespace simpla // namespace get_mesh


#endif //SIMPLA_CYLINDRICALRECTMESH_H
