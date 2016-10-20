//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H


#include <vector>
#include <iomanip>

#include "../toolbox/macro.h"
#include "../sp_def.h"
#include "../toolbox/nTuple.h"
#include "../toolbox/nTupleExt.h"
#include "../toolbox/PrettyStream.h"
#include "../toolbox/type_traits.h"
#include "../toolbox/type_cast.h"
#include "../toolbox/Log.h"


#include "RectMesh.h"


namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
struct CylindricalRectMesh : public RectMesh
{
private:
    typedef CylindricalRectMesh this_type;
    typedef Chart base_type;
public:
    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return std::string("CylindricalRectMesh"); }


public:
    static constexpr int ndims = 3;

    CylindricalRectMesh() {}

    CylindricalRectMesh(this_type const &other) : RectMesh(other)
    {

    };

    virtual  ~CylindricalRectMesh() {}


    virtual io::IOStream &save(io::IOStream &os) const
    {
//        os.open(type_cast<std::string>(this->short_id()) + "/");
//        os.set_attribute(".topology_dims", dimensions());
//        os.set_attribute(".box", box());
        return os;
    };


    point_type m_dx_;
    point_type m_inv_dx_;

    std::vector<Real> m_volume_[9];
    std::vector<Real> m_inv_volume_[9];
    std::vector<Real> m_dual_volume_[9];
    std::vector<Real> m_inv_dual_volume_[9];
public:

    vector_type const &dx() const { return m_dx_; }


    virtual point_type
    point(MeshEntityId const &s) const
    {
        UNIMPLEMENTED;
        point_type p = m::point(s);

//        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
//        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
//        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);

    }

    virtual point_type
    point_local_to_global(MeshEntityId s, point_type const &r) const
    {
        UNIMPLEMENTED;
        point_type p = m::point_local_to_global(s, r);

//        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
//        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
//        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);
    }

    virtual std::tuple<MeshEntityId, point_type>
    point_global_to_local(point_type const &g, int nId = 0) const
    {
        UNIMPLEMENTED;

        return m::point_global_to_local(point_type{0, 0, 0}, nId);
    }

    virtual index_tuple
    point_to_index(point_type const &g, int nId = 0) const
    {
        UNIMPLEMENTED;
        return m::unpack_index(std::get<0>(m::point_global_to_local(point_type{0, 0, 0}, nId)));
    };

    virtual void box(box_type const &)
    {
        UNIMPLEMENTED;
    };

    virtual box_type box(MeshEntityStatus status = SP_ES_OWNED) const
    {
        UNIMPLEMENTED;
        return box_type{};
    };

    virtual index_box_type index_box(box_type const &b) const
    {
        UNIMPLEMENTED;
        return index_box_type{};
    };

    virtual Real volume(id_type s) const { return m_volume_[m::node_id(s)][hash(s)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_[m::node_id(s)][hash(s)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_[m::node_id(s)][hash(s)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_[m::node_id(s)][hash(s)]; }

    void deploy()
    {

        RectMesh::deploy();
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
