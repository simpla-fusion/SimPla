/**
 *
 * @file rectmesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H

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

#include "MeshCommon.h"
#include "Block.h"
#include "EntityId.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
template<>
struct RectMesh : public Block
{
private:
    typedef RectMesh this_type;
    typedef Block base_type;
public:

    SP_OBJECT_HEAD(RectMesh, Block)


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

    point_type m_origin_{0, 0, 0};
    vector_type m_dx_{1, 1, 1};
    std::function<Real(Real)> m_map_[3];
public:

    RectMesh() {}

    RectMesh(RectMesh const &other) :
            Block(other),
            m_origin_(other.m_origin_),
            m_dx_(other.m_dx_) { deploy(); };

    virtual  ~RectMesh() {}

    void swap(RectMesh const &other)
    {
        std::swap(m_origin_, other.m_origin_);
        std::swap(m_dx_, other.m_dx_);
        deploy();
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
        upper = m_origin_ + m_dx_ * Block::dimensions();
        return std::make_tuple(m_origin_, upper);
    }

    vector_type const &dx() const { return m_dx_; }

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


    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s,
                                      MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }

    virtual std::shared_ptr<Chart> refine(box_type const &b, int flag = 0) const { return std::shared_ptr<Chart>(); }

private:
    vector_type m_l2g_scale_{{1, 1, 1}}, m_l2g_shift_{{0, 0, 0}};
    vector_type m_g2l_scale_{{1, 1, 1}}, m_g2l_shift_{{0, 0, 0}};

    vector_type m_inv_dx_{1, 1, 1}; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(id s) const { return m_volume_[m::node_id(s)]; }

    virtual Real dual_volume(id s) const { return m_dual_volume_[m::node_id(s)]; }

    virtual Real inv_volume(id s) const { return m_inv_volume_[m::node_id(s)]; }

    virtual Real inv_dual_volume(id s) const { return m_inv_dual_volume_[m::node_id(s)]; }


}; // struct  Mesh

void RectMesh::deploy()
{
    Block::deploy();
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
//                           MeshEntityStatus status = SP_ES_ALL) const
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
//virtual box_type box(MeshEntityStatus status = SP_ES_OWNED) const
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
//
//virtual EntityRange range(MeshEntityType entityType = VERTEX, MeshEntityStatus status = SP_ES_OWNED) const
//{
//    EntityRange res;
//
//    /**
//     *   |<-----------------------------     valid   --------------------------------->|
//     *   |<- not owned  ->|<-------------------       owned     ---------------------->|
//     *   |----------------*----------------*---*---------------------------------------|
//     *   |<---- ghost --->|                |   |                                       |
//     *   |<------------ shared  ---------->|<--+--------  not shared  ---------------->|
//     *   |<------------- DMZ    -------------->|<----------   not DMZ   -------------->|
//     *
//     */
//    switch (status)
//    {
//        case SP_ES_ALL : //all valid
//            res.append(MeshEntityIdCoder::make_range(m_outer_lower_, m_outer_upper_, entityType));
//            break;
//        case SP_ES_NON_LOCAL : // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
//        case SP_ES_SHARED : //       = 0x04,                    0b000100 shared by two or more get_mesh grid_dims
//            break;
//        case SP_ES_NOT_SHARED  : // = 0x08, //                       0b001000 not shared by other get_mesh grid_dims
//            break;
//        case SP_ES_GHOST : // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
//            res.append(
//                    MeshEntityIdCoder::make_range(
//                            index_tuple{m_outer_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                            index_tuple{m_origin_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
//            res.append(
//                    MeshEntityIdCoder::make_range(
//                            index_tuple{m_upper_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                            index_tuple{m_outer_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
//
//            if (m_dims_[1] > 1)
//            {
//                res.append(
//                        MeshEntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_origin_[1], m_outer_upper_[2]}, entityType));
//                res.append(
//                        MeshEntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_upper_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
//            }
//            if (m_dims_[2] > 1)
//            {
//                res.append(
//                        MeshEntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_origin_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_upper_[1], m_origin_[2]}, entityType));
//                res.append(
//                        MeshEntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_origin_[1], m_upper_[2]},
//                                index_tuple{m_upper_[0], m_upper_[1], m_outer_upper_[2]}, entityType));
//            }
//            break;
//        case SP_ES_DMZ: //  = 0x100,
//        case SP_ES_NOT_DMZ: //  = 0x200,
//        case SP_ES_LOCAL : // = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
//            res.append(MeshEntityIdCoder::make_range(m_inner_lower_, m_inner_upper_, entityType));
//            break;
//        case SP_ES_VALID:
//            index_tuple l, u;
//            l = m_outer_lower_;
//            u = m_outer_upper_;
//            for (int i = 0; i < 3; ++i)
//            {
//                if (m_dims_[i] > 1 && m_ghost_width_[i] != 0)
//                {
//                    l[i] += 1;
//                    u[i] -= 1;
//                }
//            }
//            res.append(MeshEntityIdCoder::make_range(l, u, entityType));
//            break;
//        case SP_ES_OWNED:
//            res.append(MeshEntityIdCoder::make_range(m_origin_, m_upper_, entityType));
//            break;
//        case SP_ES_INTERFACE: //  = 0x010, //                        0b010000 interface(boundary) shared by two get_mesh grid_dims,
//            res.append(m_interface_entities_[entityType]);
//            break;
//        default:
//            UNIMPLEMENTED;
//            break;
//    }
//    return std::move(res);
//};

//    int get_vertices(int node_id, id_type s, point_type *p = nullptr) const
//    {
//
//        int num = m::get_adjacent_entities(VERTEX, node_id, s);
//
//        if (p != nullptr)
//        {
//            id_type neighbour[num];
//
//            m::get_adjacent_entities(VERTEX, node_id, s, neighbour);
//
//            for (int i = 0; i < num; ++i)
//            {
//                p[i] = point(neighbour[i]);
//            }
//
//        }
//
//
//        return num;
//    }

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
//    virtual point_type point(id_type const &s) const { return std::move(map(m::point(s))); }
//
//    virtual point_type point_local_to_global(id_type s, point_type const &x) const
//    {
//        return std::move(map(m::point_local_to_global(s, x)));
//    }
//
//    virtual point_type point_local_to_global(std::tuple<id_type, point_type> const &t) const
//    {
//        return std::move(map(m::point_local_to_global(t)));
//    }
//
//    virtual std::tuple<id_type, point_type> point_global_to_local(point_type const &x, int n_id = 0) const
//    {
//        return std::move(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual id_type id(point_type const &x, int n_id = 0) const
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


}} // namespace simpla // namespace  mesh

#endif //SIMPLA_RECTMESH_H
