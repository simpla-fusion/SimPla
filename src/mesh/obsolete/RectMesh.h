/**
 * @file rect_mesh.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_RECT_MESH_H
#define SIMPLA_RECT_MESH_H

#include <limits>
#include "../Mesh.h"
#include "../MeshEntityIdCoder.h"
#include "MeshBlock.h"
#include "LinearMap.h"
#include "../../gtl/design_pattern/singleton_holder.h"
#include "../../gtl/utilities/memory_pool.h"
#include "../../geometry/GeoAlgorithm.h"

namespace simpla { namespace mesh
{

namespace tags { struct rect_linear; }

template<typename TMetric> using RectMesh=Mesh<TMetric, tags::rect_linear>;

/**
 * @ingroup mesh
 *
 * @brief non-Uniform structured mesh
 */
template<typename TMetric>
struct Mesh<TMetric, tags::rect_linear> : public TMetric, public MeshBlock
{

private:
    typedef Mesh<TMetric, tags::rect_linear> this_type;

    typedef MeshBlock m;

    typedef m base_type;

public:


    typedef TMetric metric_type;
    typedef MeshBlock m;

    using m::ndims;
    using m::id_type;
    using m::id_tuple;
    using m::index_type;
    using m::index_tuple;
    using m::range_type;
    using m::difference_type;

    using typename metric_type::scalar_type;
    using typename metric_type::point_type;
    using typename metric_type::vector_type;
    typedef std::tuple<point_type, point_type> box_type;

private:
    point_type m_coords_lower_ = {0, 0, 0};

    point_type m_coords_upper_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1


public:


    Mesh() : m() { }


    Mesh(this_type const &other) = delete;

    virtual  ~Mesh() { }

    this_type &operator=(this_type const &other) = delete;


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << "\tGeometry={" << std::endl
        << std::setw(indent) << "\t\t Topology = { Type = \"RectMesh\",  }," << std::endl
        << std::setw(indent) << "\t\t Box = {" << box() << "}," << std::endl
        << std::setw(indent) << "\t\t Dimensions = " << m::dimensions() << "," << std::endl
        << std::setw(indent) << "\t\t}, " << std::endl
        << std::setw(indent) << "\t}" << std::endl;

        return os;
    }

    bool is_valid() const { return m_is_valid_; }

    //================================================================================================
    // @name Coordinates dependent
    //    point_type epsilon() const { return EPSILON * m_from_topology_scale_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_lower_ = x0;
        m_coords_upper_ = x1;
    }

    void box(box_type const &b) { std::tie(m_coords_lower_, m_coords_upper_) = b; }

    box_type box() const { return (std::make_tuple(m_coords_lower_, m_coords_upper_)); }

    box_type box(id_type const &s) const
    {
        return std::make_tuple(point(s - m::_DA), point(s + m::_DA));
    };

    box_type local_box() const
    {
        point_type l_min, l_max;

        l_min = traits::get<0>(m::local_index_box());

        l_max = traits::get<1>(m::local_index_box());


        l_min = inv_map(l_min);

        l_max = inv_map(l_max);

        return (std::make_tuple(l_min, l_max));
    }


    constexpr auto dx() const DECL_RET_TYPE(m_dx_);

    index_box_type index_box(box_type const &b) const
    {

        point_type b0, b1, x0, x1;

        b0 = traits::get<0>(m::local_index_box());

        b1 = traits::get<1>(m::local_index_box());

        x0 = traits::get<0>(b);

        x1 = traits::get<1>(b);

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(
                    m::unpack_index(id(x0)),
                    m::unpack_index(id(x1) + (m::_DA << 1)));

        }
        else
        {
            index_tuple i0, i1;
            i0 = 0;
            i1 = 0;
            return std::make_tuple(i0, i1);
        }


    }

private:
    using map_type::map;
    using map_type::inv_map;
public:
    template<typename ...Args>
    point_type point(Args &&...args) const
    {
        return std::move(map_type::inv_map(m::point(std::forward<Args>(args)...)));
    }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(inv_map(m::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(m::coordinates_global_to_local(map(x), n_id));
    }

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(m::coordinates_global_to_local(map(x), n_id));
    }



    //===================================
    //
private:
    size_t hash_(id_type s) const
    {
        return m::hash(s & (~m::_DA)) * m::NUM_OF_NODE_ID +
               m::node_id(s);
    }

public:
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

//    virtual point_type const &vertex(id_type s) const
//    {
//        return m_vertices_.get()[base_type::hash(s) * base_type::NUM_OF_NODE_ID + base_type::sub_index(s)];
//    }


private:
    std::shared_ptr<Real> m_volume_;
    std::shared_ptr<Real> m_dual_volume_;
    std::shared_ptr<Real> m_inv_volume_;
    std::shared_ptr<Real> m_inv_dual_volume_;
//    std::shared_ptr<point_type> m_vertices_;
public:

    virtual void deploy();

    template<typename TGeo> void update_volume(TGeo const &geo);


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

};//struct RectMesh

template<typename TMetric, typename TMap>
void Mesh<TMetric, tags::rect_linear, TMap>::deploy()
{


    if (properties().click() > this->click())
    {
//        auto b = properties()["Geometry"]["Box"].template as<box_type>();

        std::tie(m_coords_min_, m_coords_max_) = properties()["Geometry"]["Box"].template as<box_type>();


    }

    block_type::deploy();

    auto dims = block_type::dimensions();

    map_type::set(box(), block_type::index_box(), dims);

    for (int i = 0; i < ndims; ++i)
    {
        ASSERT(dims[i] >= 1);

        ASSERT((m_coords_max_[i] - m_coords_min_[i] > EPSILON));

        m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i]) / static_cast<Real>(dims[i]);
    }

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



    m_is_valid_ = true;

    // update volume

    auto memory_block_range = block_type::template make_range<VERTEX>(block_type::memory_index_box());

    size_t num = memory_block_range.size() * block_type::NUM_OF_NODE_ID;


    m_volume_ = sp_alloc_array<Real>(num);
    m_inv_volume_ = sp_alloc_array<Real>(num);
    m_dual_volume_ = sp_alloc_array<Real>(num);
    m_inv_dual_volume_ = sp_alloc_array<Real>(num);


    parallel::parallel_for(memory_block_range, [&](range_type const &range)
    {
        for (auto const &s:range)
        {
            size_t n = this->hash(s) * block_type::NUM_OF_NODE_ID;

            block_type::get_element_volume_in_cell(*this, s, m_volume_.get() + n, m_inv_volume_.get() + n,
                                                   m_dual_volume_.get() + n, m_inv_dual_volume_.get() + n);
        }

    });


    touch();

}

}}  // namespace mesh // namespace simpla

#endif //SIMPLA_RECT_MESH_H
