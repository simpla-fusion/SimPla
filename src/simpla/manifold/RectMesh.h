/**
 *
 * @file rectmesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H

#include <simpla/SIMPLA_config.h>

#include <vector>
#include <iomanip>

#include <simpla/toolbox/macro.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/nTupleExt.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/type_cast.h>
#include <simpla/toolbox/Log.h>
#include <simpla/data/DataBlock.h>
#include <simpla/data/Attribute.h>
#include <simpla/data/DataEntityNDArray.h>
#include "../mesh/MeshCommon.h"
#include "../mesh/MeshBlock.h"
#include "../mesh/EntityId.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
struct RectMesh : public MeshBlock
{
public:

    SP_OBJECT_HEAD(RectMesh, MeshBlock)

    typedef Real scalar_type;


    /**
     *
     *   -----------------------------5
     *   |                            |
     *   |     ---------------4       |
     *   |     |              |       |
     *   |     |  ********3   |       |
     *   |     |  *       *   |       |
     *   |     |  *       *   |       |
     *   |     |  *       *   |       |
     *   |     |  2********   |       |
     *   |     1---------------       |
     *   0-----------------------------
     *
     *	5-0 = dimensions
     *	4-1 = e-d = ghosts
     *	2-1 = counts
     *
     *	0 = id_begin
     *	5 = id_end
     *
     *	1 = id_local_outer_begin
     *	4 = id_local_outer_end
     *
     *	2 = id_local_inner_begin
     *	3 = id_local_inner_end
     *
     *
     */

    point_type m_origin_{{0, 0, 0}};
    vector_type m_dx_{{1, 1, 1}};
public:
    template<typename TV, size_type IFORM> using patch_type =  data::DataEntityNDArray<TV, ndims + 1>;
    template<typename TV, size_type IFORM> using attribute_type =
    data::Attribute<data::DataEntityNDArray<TV,ndims + 1>, this_type, index_const<IFORM> >;

    RectMesh() {}

    RectMesh(RectMesh const &other) :
            MeshBlock(other),
            m_origin_(other.m_origin_),
            m_dx_(other.m_dx_) { deploy(); };

    virtual  ~RectMesh() {}

    void swap(RectMesh &other)
    {
        std::swap(m_origin_, other.m_origin_);
        std::swap(m_dx_, other.m_dx_);
        base_type::swap(other);
    }

    virtual void deploy();

    inline void box(point_type const &x0, point_type const &x1)
    {
        m_origin_ = x0;
        m_dx_ = x1 - x0;
    }

    inline void box(box_type const &b) { box(std::get<0>(b), std::get<1>(b)); }

    box_type dx() const
    {
        point_type upper;
        upper = m_origin_ + m_dx_ * MeshBlock::dimensions();
        return std::make_tuple(m_origin_, upper);
    }


    virtual point_type
    point(MeshEntityId const &s) const
    {
        point_type p = m::point(s);

        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);

    }

    virtual point_type
    point_local_to_global(MeshEntityId s, point_type const &r) const
    {
        point_type p = m::point_local_to_global(s, r);

        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);
    }

    virtual std::tuple<MeshEntityId, point_type>
    point_global_to_local(point_type const &g, int nId = 0) const
    {

        return m::point_global_to_local(
                point_type{
                        std::fma(g[0], m_g2l_scale_[0], m_g2l_shift_[0]),
                        std::fma(g[1], m_g2l_scale_[1], m_g2l_shift_[1]),
                        std::fma(g[2], m_g2l_scale_[2], m_g2l_shift_[2])
                }, nId);
    }

    virtual index_tuple
    point_to_index(point_type const &g, int nId = 0) const
    {
        return m::unpack_index(std::get<0>(m::point_global_to_local(
                point_type{
                        std::fma(g[0], m_g2l_scale_[0], m_g2l_shift_[0]),
                        std::fma(g[1], m_g2l_scale_[1], m_g2l_shift_[1]),
                        std::fma(g[2], m_g2l_scale_[2], m_g2l_shift_[2])
                }, nId)));
    };


//    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s,
//                                      MeshEntityId *p = nullptr) const
//    {
//        return m::get_adjacent_entities(entity_type, entity_type, s, p);
//    }

//    virtual std::shared_ptr<MeshBlock> refine(box_type const &b, int flag = 0) const { return std::shared_ptr<MeshBlock>(); }

private:
    vector_type m_l2g_scale_{{1, 1, 1}}, m_l2g_shift_{{0, 0, 0}};
    vector_type m_g2l_scale_{{1, 1, 1}}, m_g2l_shift_{{0, 0, 0}};

    vector_type m_inv_dx_{{1, 1, 1}}; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(MeshEntityId s) const { return m_volume_[m::node_id(s)]; }

    virtual Real dual_volume(MeshEntityId s) const { return m_dual_volume_[m::node_id(s)]; }

    virtual Real inv_volume(MeshEntityId s) const { return m_inv_volume_[m::node_id(s)]; }

    virtual Real inv_dual_volume(MeshEntityId s) const { return m_inv_dual_volume_[m::node_id(s)]; }


}; // struct  Mesh

void RectMesh::deploy()
{
    MeshBlock::deploy();
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
    auto const &dims = dimensions();

    for (int i = 0; i < ndims; ++i)
    {
        for (int i = 0; i < ndims; ++i)
        {
            assert(dims[i] > 0);

            m_dx_[i] = m_dx_[i] / static_cast<Real>( dims[i]);
            m_inv_dx_[i] = (dims[i] == 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];

            m_l2g_scale_[i] = (dims[i] == 1) ? 0 : m_dx_[i];
            m_l2g_shift_[i] = m_origin_[i];

            m_g2l_scale_[i] = (dims[i] == 1) ? 0 : m_inv_dx_[i];
            m_g2l_shift_[i] = (dims[i] == 1) ? 0 : -m_origin_[i] * m_g2l_scale_[i];

        }


        m_volume_[0 /*000*/] = 1;
        m_volume_[1 /*001*/] = (dims[0] == 1) ? 1 : m_dx_[0];
        m_volume_[2 /*010*/] = (dims[1] == 1) ? 1 : m_dx_[1];
        m_volume_[4 /*100*/] = (dims[2] == 1) ? 1 : m_dx_[2];
        m_volume_[3 /*011*/] = m_volume_[1] * m_volume_[2];
        m_volume_[5 /*101*/] = m_volume_[4] * m_volume_[1];
        m_volume_[6 /*110*/] = m_volume_[4] * m_volume_[2];
        m_volume_[7 /*111*/] = m_volume_[1] * m_volume_[2] * m_volume_[4];


        m_dual_volume_[0 /*000*/] = m_volume_[7];
        m_dual_volume_[1 /*001*/] = m_volume_[6];
        m_dual_volume_[2 /*010*/] = m_volume_[5];
        m_dual_volume_[4 /*100*/] = m_volume_[3];
        m_dual_volume_[3 /*011*/] = m_volume_[4];
        m_dual_volume_[5 /*101*/] = m_volume_[2];
        m_dual_volume_[6 /*110*/] = m_volume_[1];
        m_dual_volume_[7 /*110*/] = m_volume_[0];


        m_inv_volume_[0 /*000*/] = 1;
        m_inv_volume_[1 /*001*/] = (dims[0] == 1) ? 1 : m_inv_dx_[0];
        m_inv_volume_[2 /*010*/] = (dims[1] == 1) ? 1 : m_inv_dx_[1];
        m_inv_volume_[4 /*100*/] = (dims[2] == 1) ? 1 : m_inv_dx_[2];
        m_inv_volume_[3 /*011*/] = m_inv_volume_[2] * m_inv_volume_[1];
        m_inv_volume_[5 /*101*/] = m_inv_volume_[4] * m_inv_volume_[1];
        m_inv_volume_[6 /*110*/] = m_inv_volume_[4] * m_inv_volume_[2];
        m_inv_volume_[7 /*110*/] = m_inv_volume_[1] * m_inv_volume_[2] * m_inv_volume_[4];


        m_inv_volume_[1 /*001*/] = (dims[0] == 1) ? 0 : m_inv_volume_[1];
        m_inv_volume_[2 /*010*/] = (dims[1] == 1) ? 0 : m_inv_volume_[2];
        m_inv_volume_[4 /*100*/] = (dims[2] == 1) ? 0 : m_inv_volume_[4];


        m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
        m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
        m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
        m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
        m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
        m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
        m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
        m_inv_dual_volume_[7 /*110*/] = m_inv_volume_[0];


    }
}

//typedef typename MeshEntityIdCoder::range_type block_range_type;
//
//virtual EntityRange select(box_type const &other,
//                           MeshEntityType entityType = VERTEX,
//                           MeshZoneTag status = SP_ES_ALL) const
//{
//
//    point_type c_lower, c_upper;
//    std::tie(c_lower, c_upper) = box(status);
//
//    bool overlapped = true;
//
//    for (int i = 0; i < 3; ++i)
//    {
//        c_lower[i] = std::max(c_lower[i], std::get<0>(other)[i]);
//        c_upper[i] = std::min(c_upper[i], std::get<1>(other)[i]);
//
//        if (c_lower[i] >= c_upper[i]) { overlapped = false; }
//    }
//
//    if (!overlapped)
//    {
//        return EntityRange();
//    } else
//    {
//        return EntityRange(
//                MeshEntityIdCoder::make_range(point_to_index(c_lower), point_to_index(c_upper), entityType));
//    }
//
//};
//
//virtual box_type box(MeshZoneTag status = SP_ES_OWNED) const
//{
//    box_type res;
//
//    switch (status)
//    {
//        case SP_ES_ALL : //all valid
//            std::get<0>(res) = m_coords_lower_ - m_dx_ * m_ghost_width_;
//            std::get<1>(res) = m_coords_upper_ + m_dx_ * m_ghost_width_;;
//            break;
//        case SP_ES_LOCAL : //local and valid
//            std::get<0>(res) = m_coords_lower_ + m_dx_ * m_ghost_width_;;
//            std::get<1>(res) = m_coords_upper_ - m_dx_ * m_ghost_width_;
//            break;
//        case SP_ES_OWNED:
//            std::get<0>(res) = m_coords_lower_;
//            std::get<1>(res) = m_coords_upper_;
//            break;
//        case SP_ES_INTERFACE: //SP_ES_INTERFACE
//        case SP_ES_GHOST : //local and valid
//        default:
//            UNIMPLEMENTED;
//            break;
//
//
//    }
//    return std::move(res);
//}
//
//
//virtual EntityRange range(box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return range(index_box(b), entityType);
//}
//
//virtual EntityRange range(index_box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return MeshEntityIdCoder::make_range(b, entityType);
//}

/**
 * @name  Coordinate map
 * @{
 *
 *        Topology mesh       geometry get_mesh
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
//private:
//
//
//    point_type m_l2g_shift_ = {0, 0, 0};
//
//    point_type m_l2g_scale_ = {1, 1, 1};
//
//    point_type m_g2l_shift_ = {0, 0, 0};
//
//    point_type m_g2l_scale_ = {1, 1, 1};
//
//
//    point_type inv_map(point_type const &x) const
//    {
//
//        point_type res;
//
//        res[0] = std::fma(x[0], m_g2l_scale_[0], m_g2l_shift_[0]);
//
//        res[1] = std::fma(x[1], m_g2l_scale_[1], m_g2l_shift_[1]);
//
//        res[2] = std::fma(x[2], m_g2l_scale_[2], m_g2l_shift_[2]);
//
//        return std::move(res);
//    }
//
//    point_type map(point_type const &y) const
//    {
//
//        point_type res;
//
//
//        res[0] = std::fma(y[0], m_l2g_scale_[0], m_l2g_shift_[0]);
//
//        res[1] = std::fma(y[1], m_l2g_scale_[1], m_l2g_shift_[1]);
//
//        res[2] = std::fma(y[2], m_l2g_scale_[2], m_l2g_shift_[2]);
//
//        return std::move(res);
//    }
//
//public:
//
//    virtual point_type point(mesh_id_type const &s) const { return std::move(map(m::point(s))); }
//
//    virtual point_type point_local_to_global(mesh_id_type s, point_type const &x) const
//    {
//        return std::move(map(m::point_local_to_global(s, x)));
//    }
//
//    virtual point_type point_local_to_global(std::tuple<mesh_id_type, point_type> const &t) const
//    {
//        return std::move(map(m::point_local_to_global(t)));
//    }
//
//    virtual std::tuple<mesh_id_type, point_type> point_global_to_local(point_type const &x, int n_id = 0) const
//    {
//        return std::move(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual mesh_id_type id(point_type const &x, int n_id = 0) const
//    {
//        return std::get<0>(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//
//
//    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
//    {
//
//        point_type b0, b1, x0, x1;
//
//        std::tie(b0, b1) = local_index_box();
//
//        std::tie(x0, x1) = b;
//
//        if (geometry::box_intersection(b0, b1, &x0, &x1))
//        {
//            return std::make_tuple(m::unpack_index(id(x0)),
//                                   m::unpack_index(id(x1) + (m::_DA << 1)));
//
//        }
//        else
//        {
//            index_tuple i0, i1;
//            i0 = 0;
//            i1 = 0;
//            return std::make_tuple(i0, i1);
//        }
//
//    }


//    struct calculus_policy
//    {
//        template<typename ...Args> static double eval(Args &&...args) { return 1.0; }
//    };


}} // namespace simpla // namespace  mesh_as

#endif //SIMPLA_RECTMESH_H
