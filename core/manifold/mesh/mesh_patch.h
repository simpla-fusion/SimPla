/**
 * @file mesh_patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_MESH_PATCH_H
#define SIMPLA_MESH_PATCH_H

#include <tuple>
#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"

namespace simpla { namespace mesh
{
/**
 *
 * @ingroup mesh
 **/
template<typename ...> struct Patch;

template<typename TBaseMesh>
struct Patch
{
    typedef TBaseMesh base_mesh;
    base_mesh const &m_base_;
    typedef typename base_mesh::point_type point_type;
    typedef typename base_mesh::id_type id_type;


    std::tuple<point_type, point_type> box() const;

    std::tuple<point_type, point_type> local_box() const;

    constexpr auto dx() const DECL_RET_TYPE(m_base_.dx());

    template<typename ...Args>
    point_type point(Args &&...args) const
    {
        return base_mesh::point(std::forward<Args>(args)...);
    }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(map(base_type::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(base_type::coordinates_global_to_local(inv_map(x), n_id));
    }

    virtual id_type id(point_type x, int n_id = 0) const
    {
        return std::get<0>(base_type::coordinates_global_to_local(inv_map(x), n_id));
    }

    //===================================
    //

    virtual Real volume(id_type s) const
    {
        return m_volume_.get()[hash_(s)];
    }

    virtual Real dual_volume(id_type s) const
    {
        return m_dual_volume_.get()[hash_(s)];
    }

    virtual Real inv_volume(id_type s) const
    {

        return m_inv_volume_.get()[hash_(s)];
    }

    virtual Real inv_dual_volume(id_type s) const
    {
        return m_inv_dual_volume_.get()[hash_(s)];
    }

    virtual point_type const &vertex(id_type s) const
    {
        return m_vertices_.get()[base_type::hash(s) * base_type::NUM_OF_NODE_ID + base_type::sub_index(s)];
    }
};

}//namespace mesh
}//namespace simpla

#endif //SIMPLA_MESH_PATCH_H
