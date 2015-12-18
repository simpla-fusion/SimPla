/**
 * @file rect_mesh.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_RECT_MESH_H
#define SIMPLA_RECT_MESH_H

#include <limits>
#include "Mesh.h"
#include "MeshIds.h"
#include "MeshBlock.h"
#include "LinearMap.h"
#include "../../gtl/design_pattern/singleton_holder.h"
#include "../../gtl/utilities/memory_pool.h"
#include "../../geometry/geo_algorithm.h"

namespace simpla { namespace mesh
{

namespace tags { struct rect_linear; }

template<typename TMetric> using RectMesh=Mesh<TMetric, tags::rect_linear, LinearMap>;

/**
 * @ingroup mesh
 *
 * @brief non-Uniform structured mesh
 */
template<typename TMetric, typename TMap>
struct Mesh<TMetric, tags::rect_linear, TMap> : public TMetric, public MeshBlock, public TMap
{

private:
    typedef Mesh<TMetric, tags::rect_linear, TMap> this_type;
    typedef TMap map_type;
    typedef TMetric metric_type;

    typedef MeshBlock block_type;

public:
    using block_type::ndims;
    using block_type::id_type;
    using block_type::id_tuple;
    using block_type::index_type;
    using block_type::index_tuple;
    using block_type::range_type;
    using block_type::point_type;
    using block_type::vector_type;
    using block_type::difference_type;

private:
    point_type m_coords_min_ = {0, 0, 0};

    point_type m_coords_max_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

    bool m_is_valid_ = false;
public:


    Mesh() : block_type() { }


    Mesh(this_type const &other) : block_type(other), m_dx_(other.m_dx_) { }

    virtual  ~ Mesh() { }

    virtual void swap(this_type &other)
    {

        block_type::swap(other);

        map_type::swap(other);

        std::swap(m_coords_min_, other.m_coords_min_);
        std::swap(m_coords_max_, other.m_coords_max_);
        std::swap(m_dx_, other.m_dx_);


        deploy();
        other.deploy();

    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    static std::string topology_type() { return "SMesh"; }


    template<typename TDict>
    void load(TDict const &dict)
    {
        box(dict["Geometry"]["Box"].template as<std::tuple<point_type, point_type> >());

        block_type::dimensions(
                dict["Geometry"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1}));
    }

    virtual std::ostream &print(std::ostream &os) const
    {

        os
        << "\tgeometry={" << std::endl
        << "\t\t Topology = { Type = \"RectMesh\",  }," << std::endl
        << "\t\t Box = {" << box() << "}," << std::endl
        << "\t\t Dimensions = " << block_type::dimensions() << "," << std::endl
        << "\t\t}, "
        << "\t}"
        << std::endl;

        return os;
    }

    bool is_valid() const { return m_is_valid_; }

    //================================================================================================
    // @name Coordinates dependent
    //    point_type epsilon() const { return EPSILON * m_from_topology_scale_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_min_ = x0;
        m_coords_max_ = x1;
    }

    template<typename T0> void box(T0 const &b)
    {
        box(simpla::traits::get<0>(b), simpla::traits::get<1>(b));
    }


    std::tuple<point_type, point_type> box() const
    {
        return (std::make_tuple(m_coords_min_, m_coords_max_));
    }

    std::tuple<point_type, point_type> box(id_type const &s) const
    {
        return std::make_tuple(point(s - block_type::_DA), point(s + block_type::_DA));
    };

    std::tuple<point_type, point_type> local_box() const
    {
        point_type l_min, l_max;

        std::tie(l_min, l_max) = block_type::local_box();

        l_min = inv_map(l_min);

        l_max = inv_map(l_max);

        return (std::make_tuple(l_min, l_max));
    }


    constexpr auto dx() const DECL_RET_TYPE(m_dx_);

    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
    {

        point_type b0, b1, x0, x1;

        std::tie(b0, b1) = local_box();
        std::tie(x0, x1) = b;

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(block_type::unpack_index(id(x0)),
                                   block_type::unpack_index(id(x1) + (block_type::_DA << 1)));

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
        return std::move(map_type::inv_map(block_type::point(std::forward<Args>(args)...)));
    }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(inv_map(block_type::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(block_type::coordinates_global_to_local(map(x), n_id));
    }

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(block_type::coordinates_global_to_local(map(x), n_id));
    }



    //===================================
    //
private:
    size_t hash_(id_type s) const
    {
        return block_type::hash(s & (~block_type::_DA)) * block_type::NUM_OF_NODE_ID +
               block_type::node_id(s);
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
    block_type::deploy();

    auto dims = block_type::dimensions();

    map_type::set(box(), block_type::box(), dims);

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


}

}}  // namespace mesh // namespace simpla

#endif //SIMPLA_RECT_MESH_H
