/**
 * @file rect_mesh.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_RECT_MESH_H
#define SIMPLA_RECT_MESH_H

#include "corectmesh.h"

namespace simpla { namespace topology
{


struct RectMesh : public CoRectMesh
{
private:
    typedef RectMesh this_type;
    typedef CoRectMesh m;
public:

    RectMesh() : m()
    {
    }

    RectMesh(this_type const &other) : m(other) { }

    virtual  ~RectMesh() { }

    virtual void swap(this_type &other) { m::swap(other); }

    template<typename OS>
    OS &print(OS &os) const
    {

        os << "\t\tTopology = {" << std::endl
        << "\t\t Type = \"RectMesh\"," << std::endl
        << "\t\t Extents = {" << box() << "}," << std::endl
        << "\t\t Count = {}," << std::endl
        << "\t\t}, " << std::endl;

        return os;
    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }
//
//    virtual point_type point(id_type const &s) const { return std::move(map(m::coordinates(s))); }
//
//    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
//    {
//        return std::move(map(m::coordinates_local_to_global(t)));
//    }
//
//    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
//    {
//        return std::move(m::coordinates_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual id_type id(point_type x, int n_id = 0) const
//    {
//        return std::get<0>(m::coordinates_global_to_local(inv_map(x), n_id));
//    }

    virtual Real volume(id_type s) const { return m_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual point_type const &vertex(id_type s) const { return m_vertics_.get()[hash(s & (~m::_DA))]; }

    virtual void deploy() { m::deploy(); }


    std::shared_ptr<Real> m_volume_;
    std::shared_ptr<Real> m_dual_volume_;
    std::shared_ptr<Real> m_inv_volume_;
    std::shared_ptr<Real> m_inv_dual_volume_;
    std::shared_ptr<point_type> m_vertics_;


};//struct RectMesh

}}  // namespace topology // namespace simpla

#endif //SIMPLA_RECT_MESH_H
