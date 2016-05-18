/**
 * @file MeshBlockStructured.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHBLOCKSTRUCTURED_H
#define SIMPLA_MESH_MESHBLOCKSTRUCTURED_H

#include "../base/Object.h"
#include "../gtl/IteratorBlock.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshIds.h"

namespace simpla { namespace tags { struct split; }}

namespace simpla { namespace mesh
{
namespace tags { template<int> class BlockStructured; }
/**
 *  MeshBlockStructured : Block Structured Mesh
 *  - topology  : rectangle or hexahedron;
 *  - cell size : uniform structured;
 *  - decompose, refine,coarse => Block Structured Mesh
 */
template<int NDIMS>
class Mesh<tags::BlockStructured<NDIMS>> : public base::Object
{
    typedef Mesh<tags::BlockStructured<NDIMS>> this_type;
public:

// types
    static constexpr int ndims = NDIMS;
    typedef size_t size_type;
    typedef gtl::IteratorBlock<size_type, ndims> iterator;



    //****************************************
    // Range concept

    Mesh(iterator b, iterator e, size_type grain_size = 1) :
            m_begin_(*b), m_end_(*e), m_grain_size_(grain_size)
    {
    }

    Mesh(this_type &r, tags::split) : m_begin_(r.m_begin_ + r.size() / 2), m_end_(r.m_end_),
                                      m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    Mesh(this_type &r, tags::proportional_split &proportion) :
            m_begin_(r.m_begin_ + r.size() * proportion.left() / (proportion.left() + proportion.
                    right())),
            m_end_(r.m_end_),
            m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    // Proportional split is enabled
    static const bool is_splittable_in_proportion = true;

    // capacity
    size_type size() const { return traits::distance(m_begin_, m_end_); };

    bool empty() const { return m_begin_ == m_end_; };

    // access
    size_type grainsize() const { return m_grain_size_; }

    bool is_divisible() const { return size() > grainsize(); }

    // iterators
    iterator begin() const { return m_begin_; }

    iterator end() const { return m_end_; }

    //****************************************

    int number_of_dims() const { return m_ndims_; }

    index_tuple const &dimensions() const { return m_dimensions_; }

    index_tuple const &ghost_width() const { return m_ghost_width_; }


    bool is_periodic(int n) const { return m_ghost_width_[n % 3] == 0; }

    /**
     *  remove periodic axis, which  ghost_width==0
     */
    mesh_entity_id_t periodic_id_mask() const
    {
        mesh_entity_id_t M0 = ((1UL << ID_DIGITS) - 1);
        mesh_entity_id_t M1 = ((1UL << (MESH_RESOLUTION)) - 1);
        return FULL_OVERFLOW_FLAG
               | (is_periodic(0) ? M1 : M0)
               | ((is_periodic(1) ? M1 : M0) << ID_DIGITS)
               | ((is_periodic(2) ? M1 : M0) << (ID_DIGITS * 2));
    }

    size_t id_mask() const
    {
        mesh_entity_id_t M0 = ((1UL << ID_DIGITS) - 1);
        mesh_entity_id_t M1 = ((1UL << (MESH_RESOLUTION)) - 1);
        return FULL_OVERFLOW_FLAG
               | ((m_idx_max_[0] - m_idx_min_[0] > 1) ? M0 : M1)
               | (((m_idx_max_[1] - m_idx_min_[1] > 1) ? M0 : M1) << ID_DIGITS)
               | (((m_idx_max_[2] - m_idx_min_[2] > 1) ? M0 : M1) << (ID_DIGITS * 2));
    }


    index_box_type index_box() const
    {
        return std::make_tuple(m_idx_min_, m_idx_max_);
    }


//    index_box_type cell_index_box(mesh_entity_id_t const &s) const
//    {
//        return std::make_tuple(m::unpack_index(s - _DA), m::unpack_index(s + _DA));
//    }
    index_box_type index_box(box_type const &b) const
    {

        point_type b0, b1, x0, x1;

        std::tie(b0, b1) = local_index_box();

        std::tie(x0, x1) = b;

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(m::unpack_index(id(x0)),
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

    index_box_type local_index_box() const
    {
        return (std::make_tuple(m_idx_local_min_, m_idx_local_max_));
    }

    index_box_type memory_index_box() const
    {
        return (std::make_tuple(m_idx_memory_min_, m_idx_memory_max_));
    }

    bool in_box(index_tuple const &x) const
    {
        return (m_idx_local_min_[1] <= x[1]) && (m_idx_local_min_[2] <= x[2]) && (m_idx_local_min_[0] <= x[0])  //
               && (m_idx_local_max_[1] > x[1]) && (m_idx_local_max_[2] > x[2]) && (m_idx_local_max_[0] > x[0]);

    }

    bool in_box(mesh_entity_id_t s) const { return in_box(m::unpack_index(s)); }

    template<int IFORM>
    range_type range() const
    {
        return m::template make_range<IFORM>(m_idx_local_min_, m_idx_local_max_);
    }

    template<int IFORM>
    range_type inner_range() const
    {
        index_tuple i_min = m_idx_local_min_;
        index_tuple i_max = m_idx_local_max_;
        for (int n = 0; n < ndims; ++n)
        {
            if (i_max[n] - i_min[n] > 2 * m_idx_min_[n])
            {
                i_min[n] += m_idx_min_[n];
                i_max[n] -= m_idx_min_[n];
            }

        }
        return m::template make_range<IFORM>(i_min, i_max);
    }

    template<int IFORM>
    range_type outer_range() const
    {
        return m::template make_range<IFORM>(m_idx_memory_min_, m_idx_memory_max_);
    }


    template<int IFORM>
    size_t max_hash() const
    {
        return static_cast<size_t>(m::hash(
                m::pack_index(m_idx_memory_max_ - 1, m::template sub_index_to_id<IFORM>(3UL)),
                m_idx_memory_min_, m_idx_memory_max_));
    }


    size_t hash(mesh_entity_id_t const &s) const
    {
        return static_cast<size_t>(m::hash(s, m_idx_memory_min_, m_idx_memory_max_));
    }


    index_box_type const &center_box() const { return m_center_box_; }

    std::vector<index_box_type> const &boundary_box() const { return m_boundary_box_; }

    std::vector<index_box_type> const &ghost_box() const { return m_ghost_box_; }




    //================================================================================================
    // @name Coordinates dependent
private:
    point_type m_min_;
    point_type m_max_;
    vector_type m_dx_;
public:

    void box(box_type const &b) { std::tie(m_min_, m_max_) = b; }

    box_type box() const { return (std::make_tuple(m_min_, m_max_)); }

    vector_type const &dx() const { return m_dx_; }

    box_type cell_box(mesh_entity_id_t const &s) const
    {
        return std::make_tuple(point(s - m::_DA), point(s + m::_DA));
    }

    int get_vertices(size_t node_id, mesh_entity_id_t s, point_type *p = nullptr) const
    {

        int num = m::get_adjacent_cells(VERTEX, node_id, s);

        if (p != nullptr)
        {
            mesh_entity_id_t neighbour[num];

            m::get_adjacent_cells(VERTEX, node_id, s, neighbour);

            for (int i = 0; i < num; ++i)
            {
                p[i] = point(neighbour[i]);
            }

        }

        return num;
    }

    void get_volumes(Real *v, Real *inv_v, Real *dual_v, Real *inv_dual_v);

private:
    /**
     * @name  Coordinate map
     * @{
     *
     *        Topology mesh       geometry mesh
     *                        map
     *              M      ---------->      G
     *              x                       y
     **/

    point_type m_map_orig_ = {0, 0, 0};

    point_type m_map_scale_ = {1, 1, 1};

    point_type m_inv_map_orig_ = {0, 0, 0};

    point_type m_inv_map_scale_ = {1, 1, 1};


    point_type inv_map(point_type const &x) const
    {

        point_type res;

        res[0] = std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);

        res[1] = std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);

        res[2] = std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);

        return std::move(res);
    }

    point_type map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);

        res[1] = std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);

        res[2] = std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);

        return std::move(res);
    }

public:

    virtual point_type point(mesh_entity_id_t const &s) const
    {
        return std::move(map(m::point(s)));
    }

    virtual point_type coordinates_local_to_global(mesh_entity_id_t s, point_type const &x) const
    {
        return std::move(map(m::coordinates_local_to_global(s, x)));
    }

    virtual point_type coordinates_local_to_global(std::tuple<mesh_entity_id_t, point_type> const &t) const
    {
        return std::move(map(m::coordinates_local_to_global(t)));
    }

    virtual std::tuple<mesh_entity_id_t, point_type> coordinates_global_to_local(point_type const &x,
                                                                                 int n_id = 0) const
    {
        return std::move(m::coordinates_global_to_local(inv_map(x), n_id));
    }

    virtual mesh_entity_id_t id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(m::coordinates_global_to_local(inv_map(x), n_id));
    }


private:
    index_box_type m_center_box_;
    std::vector<index_box_type> m_boundary_box_;
    std::vector<index_box_type> m_ghost_box_;

};//class MeshBlockStructured

}}//namespace simpla { namespace mesh
#endif //SIMPLA_MESH_MESHBLOCKSTRUCTURED_H
