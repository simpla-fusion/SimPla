/**
 *
 * @file CoRectMesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <vector>
#include <iomanip>

#include "../mesh/DataBlock.h"
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

struct CoRectMesh : public MeshBlock
{

public:
    static constexpr unsigned int NDIMS = 3;

    SP_OBJECT_HEAD(CoRectMesh, MeshBlock)

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



public:

    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<TV, IFORM>;

    CoRectMesh() {}

    template<typename ...Args>
    CoRectMesh(Args &&...args): MeshBlock(std::forward<Args>(args)...) {};

    CoRectMesh(CoRectMesh const &other) : MeshBlock(other)
    {

        for (int i = 0; i < 9; ++i)
        {
            m_volume_[i] = m_volume_[i];
            m_inv_volume_[i] = m_inv_volume_[i];
            m_dual_volume_[i] = m_dual_volume_[i];
            m_inv_dual_volume_[i] = m_inv_dual_volume_[i];
        }


    };

    virtual  ~CoRectMesh() {}

    virtual void deploy();

    virtual std::shared_ptr<MeshBlock> clone() const { return std::make_shared<CoRectMesh>(*this); };

    template<typename TV, mesh::MeshEntityType IFORM>
    std::shared_ptr<mesh::DataBlock> create_data_block(void *p) const
    {
        auto b = outer_index_box();

        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::get<0>(b)[2], 0};
        index_type hi[4] = {std::get<1>(b)[0], std::get<1>(b)[1], std::get<0>(b)[2], 3};
        return std::dynamic_pointer_cast<mesh::DataBlock>(
                std::make_shared<data_block_type<TV, IFORM>>(
                        static_cast<TV *>(p),
                        (IFORM == mesh::VERTEX || IFORM == mesh::VOLUME) ? 3 : 4,
                        lo, hi));
    };


private:
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

void CoRectMesh::deploy()
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
    m_dual_volume_[7 /*111*/] = m_volume_[0];


    m_inv_volume_[0 /*000*/] = 1;
    m_inv_volume_[1 /*001*/] = (dims[0] == 1) ? 1 : m_inv_dx_[0];
    m_inv_volume_[2 /*010*/] = (dims[1] == 1) ? 1 : m_inv_dx_[1];
    m_inv_volume_[4 /*100*/] = (dims[2] == 1) ? 1 : m_inv_dx_[2];
    m_inv_volume_[3 /*011*/] = m_inv_volume_[2] * m_inv_volume_[1];
    m_inv_volume_[5 /*101*/] = m_inv_volume_[4] * m_inv_volume_[1];
    m_inv_volume_[6 /*110*/] = m_inv_volume_[4] * m_inv_volume_[2];
    m_inv_volume_[7 /*111*/] = m_inv_volume_[1] * m_inv_volume_[2] * m_inv_volume_[4];


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
    m_inv_dual_volume_[7 /*111*/] = m_inv_volume_[0];



}
}} // namespace simpla // namespace  mesh_as

#endif //SIMPLA_CORECTMESH_H
//typedef typename MeshEntityIdCoder::range_type block_range_type;
//
//virtual EntityIdRange select(box_type const &other,
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
//        return EntityIdRange();
//    } else
//    {
//        return EntityIdRange(
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
//virtual EntityIdRange range(box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return range(index_box(b), entityType);
//}
//
//virtual EntityIdRange range(index_box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return MeshEntityIdCoder::make_range(b, entityType);
//}
//
//virtual EntityIdRange range(MeshEntityType entityType = VERTEX, MeshZoneTag status = SP_ES_OWNED) const
//{
//    EntityIdRange res;
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

//    int get_vertices(int node_id, mesh_id_type s, point_type *p = nullptr) const
//    {
//
//        int num = m::get_adjacent_entities(VERTEX, node_id, s);
//
//        if (p != nullptr)
//        {
//            mesh_id_type neighbour[num];
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
//    point_type m_l2g_shift_ = {{0, 0, 0}};
//
//    point_type m_l2g_scale_ = {{1, 1, 1}};
//
//    point_type m_g2l_shift_ = {{0, 0, 0}};
//
//    point_type m_g2l_scale_ = {{1, 1, 1}};
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



