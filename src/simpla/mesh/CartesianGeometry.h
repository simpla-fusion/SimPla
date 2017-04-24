/**
 *
 * @file CartesianGeometry.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_CARTESIAN_GEOMETRY_H
#define SIMPLA_CARTESIAN_GEOMETRY_H

#include <iomanip>
#include <vector>
#include "CoRectMesh.h"
#include "Mesh.h"
#include "simpla/engine/all.h"
#include "simpla/utilities/sp_def.h"

namespace simpla {
namespace mesh {
struct CartesianGeometry : public engine::Chart {
    SP_OBJECT_HEAD(CartesianGeometry, engine::Chart);
    DECLARE_REGISTER_NAME("CartesianGeometry");

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto p = engine::Chart::Serialize();
        p->SetValue<std::string>("Type", ClassName());
        return p;
    };
};

/**
 * @ingroup mesh
 * @brief Uniform structured get_mesh
 */
template <>
struct Mesh<CartesianGeometry, CoRectMesh> : public CoRectMesh {
    typedef Mesh<CartesianGeometry, CoRectMesh> mesh_type;
    SP_OBJECT_HEAD(mesh_type, CoRectMesh)

   public:
    typedef Real scalar_type;
    explicit Mesh(std::shared_ptr<engine::Chart> c = nullptr)
        : CoRectMesh(c != nullptr ? c : std::make_shared<CartesianGeometry>()) {}

    ~Mesh() override = default;
    SP_DEFAULT_CONSTRUCT(Mesh);
    DECLARE_REGISTER_NAME("Mesh<CartesianGeometry,CoRectMesh>");

    void SetUp() override {
        if (GetChart() == nullptr) { CoRectMesh::SetChart(std::make_shared<CartesianGeometry>()); }
    }

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto p = CoRectMesh::Serialize();
        p->template SetValue<std::string>("Type", ClassName());
        return p;
    };
};  // struct  Mesh

}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CARTESIAN_GEOMETRY_H

// typedef typename EntityIdCoder::range_type block_range_type;
//
// virtual EntityIdRange select(box_type const &other,
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
//        c_lower[i] = std::max(c_lower[i], std::PopPatch<0>(other)[i]);
//        c_upper[i] = std::min(c_upper[i], std::PopPatch<1>(other)[i]);
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
//                EntityIdCoder::make_range(point_to_index(c_lower), point_to_index(c_upper),
//                entityType));
//    }
//
//};
//
// virtual box_type box(MeshZoneTag status = SP_ES_OWNED) const
//{
//    box_type res;
//
//    switch (status)
//    {
//        case SP_ES_ALL : //all valid
//            std::PopPatch<0>(res) = m_coords_lower_ - m_dx_ * m_ghost_width_;
//            std::PopPatch<1>(res) = m_coords_upper_ + m_dx_ * m_ghost_width_;;
//            break;
//        case SP_ES_LOCAL : //local and valid
//            std::PopPatch<0>(res) = m_coords_lower_ + m_dx_ * m_ghost_width_;;
//            std::PopPatch<1>(res) = m_coords_upper_ - m_dx_ * m_ghost_width_;
//            break;
//        case SP_ES_OWNED:
//            std::PopPatch<0>(res) = m_coords_lower_;
//            std::PopPatch<1>(res) = m_coords_upper_;
//            break;
//        case SP_ES_INTERFACE: //SP_ES_INTERFACE
//        case SP_ES_GHOST : //local and valid
//        default:
//            UNIMPLEMENTED;
//            break;
//
//
//    }
//    return std::Move(res);
//}
//
//
// virtual EntityIdRange Range(box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return Range(index_box(b), entityType);
//}
//
// virtual EntityIdRange Range(index_box_type const &b, MeshEntityType entityType = VERTEX) const
//{
//    return EntityIdCoder::make_range(b, entityType);
//}
//
// virtual EntityIdRange Range(MeshEntityType entityType = VERTEX, MeshZoneTag status = SP_ES_OWNED)
// const
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
//            res.append(EntityIdCoder::make_range(m_outer_lower_, m_outer_upper_, entityType));
//            break;
//        case SP_ES_NON_LOCAL : // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
//        case SP_ES_SHARED : //       = 0x04,                    0b000100 shared by two or more
//        get_mesh grid_dims
//            break;
//        case SP_ES_NOT_SHARED  : // = 0x08, //                       0b001000 not shared by other
//        get_mesh grid_dims
//            break;
//        case SP_ES_GHOST : // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
//            res.append(
//                    EntityIdCoder::make_range(
//                            index_tuple{m_outer_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                            index_tuple{m_origin_[0], m_outer_upper_[1], m_outer_upper_[2]},
//                            entityType));
//            res.append(
//                    EntityIdCoder::make_range(
//                            index_tuple{m_upper_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                            index_tuple{m_outer_upper_[0], m_outer_upper_[1], m_outer_upper_[2]},
//                            entityType));
//
//            if (m_dims_[1] > 1)
//            {
//                res.append(
//                        EntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_outer_lower_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_origin_[1], m_outer_upper_[2]},
//                                entityType));
//                res.append(
//                        EntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_upper_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_outer_upper_[1], m_outer_upper_[2]},
//                                entityType));
//            }
//            if (m_dims_[2] > 1)
//            {
//                res.append(
//                        EntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_origin_[1], m_outer_lower_[2]},
//                                index_tuple{m_upper_[0], m_upper_[1], m_origin_[2]}, entityType));
//                res.append(
//                        EntityIdCoder::make_range(
//                                index_tuple{m_origin_[0], m_origin_[1], m_upper_[2]},
//                                index_tuple{m_upper_[0], m_upper_[1], m_outer_upper_[2]},
//                                entityType));
//            }
//            break;
//        case SP_ES_DMZ: //  = 0x100,
//        case SP_ES_NOT_DMZ: //  = 0x200,
//        case SP_ES_LOCAL : // = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
//            res.append(EntityIdCoder::make_range(m_inner_lower_, m_inner_upper_, entityType));
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
//            res.append(EntityIdCoder::make_range(l, u, entityType));
//            break;
//        case SP_ES_OWNED:
//            res.append(EntityIdCoder::make_range(m_origin_, m_upper_, entityType));
//            break;
//        case SP_ES_INTERFACE: //  = 0x010, //                        0b010000 interface(boundary)
//        shared by two get_mesh grid_dims,
//            res.append(m_interface_entities_[entityType]);
//            break;
//        default:
//            UNIMPLEMENTED;
//            break;
//    }
//    return std::Move(res);
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
// private:
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
//        return std::Move(res);
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
//        return std::Move(res);
//    }
//
// public:
//
//    virtual point_type point(mesh_id_type const &s) const { return std::Move(map(m::point(s))); }
//
//    virtual point_type point_local_to_global(mesh_id_type s, point_type const &x) const
//    {
//        return std::Move(map(m::point_local_to_global(s, x)));
//    }
//
//    virtual point_type point_local_to_global(std::tuple<mesh_id_type, point_type> const &t) const
//    {
//        return std::Move(map(m::point_local_to_global(t)));
//    }
//
//    virtual std::tuple<mesh_id_type, point_type> point_global_to_local(point_type const &x, int
//    n_id = 0) const
//    {
//        return std::Move(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual mesh_id_type id(point_type const &x, int n_id = 0) const
//    {
//        return std::PopPatch<0>(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//
//
//    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b)
//    const
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
