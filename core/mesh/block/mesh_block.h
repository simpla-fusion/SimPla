/**
 * @file mesh_block.h
 * Created by salmon on 15-7-2.
 */

#ifndef SIMPLA_MESH_BLOCK_BLOCK_H_
#define SIMPLA_MESH_BLOCK_BLOCK_H_

class nTuple;

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/design_pattern/signal.h"


#include "../../geometry/coordinate_system.h"

#include <vector>
#include <list>
#include "mesh_id.h"
#include "mesh_block_id.h"

namespace simpla {
template<typename ...> struct MeshConnection;

namespace tags {

template<int LEVEL> struct multi_block : public std::integral_constant<int, LEVEL> { };
struct split;

}

template<typename CS, int LEVEL> using BlockMesh= Mesh<CS, tags::multi_block<LEVEL>>;


/**
 *  @brief Block represents a 'NDIMS'-dimensional 'LEVEL'th-level AMR mesh ;
 */
template<typename CS, int LEVEL>
struct Mesh<CS, tags::multi_block<LEVEL>>
{


public:

    static constexpr int ndims = geometry::traits::dimension<CS>::value;

    static constexpr int level = tags::multi_block<LEVEL>::value;


    typedef Mesh<CS, tags::multi_block<level> > this_type;

    typedef Mesh<CS, tags::multi_block<level - 1> > finer_mesh;

    typedef Mesh<CS, tags::multi_block<level + 1> > coarser_mesh;


    typedef geometry::traits::point_type_t<CS> point_type;


    typedef MeshID<ndims, level> m;


    typedef typename m::id_type id_type;

    typedef typename m::id_tuple id_tuple;

    typedef typename m::index_type index_type;
    typedef typename m::index_tuple index_tuple;
    typedef typename m::range_type range_type;

/**
 *
 *   -----------------------------5
 *   |                            |
 *   |     ---------------4       |
 *   |     |       B      |       |
 *   |     |  ********3   |       |
 *   |     |  *       *   |       |
 *   |     |  *   A   *   |       |
 *   |     |  *       *   |       |
 *   |     |  2********   |       |
 *   |     1---------------       |
 *   0-----------------------------
 *
 *  A is control region;
 *  B is affect region;
 *  B - A is ghost region
 *
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

    id_type m_id_min_;
    id_type m_id_max_;

    point_type m_coord_min_, m_coord_max_;
    index_tuple m_dimensions_;
    typedef MeshConnection<this_type> connection_type;
    std::list<connection_type> m_adjacency_list_;
public:


    Mesh(Mesh &other, tags::split) :
            m_coord_min_(other.m_coord_min_), m_coord_max_(other.m_coord_max_), m_dimensions_(other.m_dimensions_)
    {
        m_adjacency_list_.emplace_back(&other);
        other.m_adjacency_list_.emplace_back(this);
    }

    ~Mesh()
    {

    }

    Mesh(point_type const &x_min, point_type const &x_max, index_tuple const &d) :
            m_coord_min_(x_min), m_coord_max_(x_max), m_dimensions_(d)
    {

    }


    void swap(this_type &other)
    {
        std::swap(m_id_min_, other.m_id_min_);
        std::swap(m_id_max_, other.m_id_max_);

    }


    typename m::index_tuple dimensions() const
    {
        return m::unpack_index(m_id_max_ - m_id_min_);
    }

    size_t ghost_width() const
    {

    };

    template<size_t IFORM>
    size_t max_hash() const
    {
        return m::template max_hash<IFORM>(m_id_min_, m_id_max_);
    }

    size_t hash(id_type s) const
    {
        return m::hash(s, m_id_min_, m_id_max_);
    }


    std::tuple<index_tuple, index_tuple> index_box() const
    {
        return std::make_tuple(m::unpack_index(m_id_min_),
                               m::unpack_index(m_id_max_));
    }


    auto box() const
    DECL_RET_TYPE(std::forward_as_tuple(m_coord_min_, m_coord_max_))


    template<typename ...Args>
    range_type range(Args &&...args) const
    {
        return m::range_type(m_id_min_, m_id_max_, std::forward<Args>(args)...);
    }


public:


    Signal<void(id_type)> signal_finer;
    Signal<void(id_type)> signal_coarse;
    Signal<void(id_type)> signal_update;
    Signal<void(id_type)> signal_sync;


    template<typename T, typename ...Args>
    std::shared_ptr<T> create(Args &&...args)
    {
        auto res = std::make_shared<T>(*this, std::forward<Args>(args)...);
        signal_finer.connect(res, &T::finer);
        signal_coarse.connect(res, &T::coarse);
        signal_update.connect(res, &T::update);
        signal_update.connect(sync, &T::sync);

        return res;
    }

    BlockID add_finer_mesh(index_tuple const &imin, index_tuple const &imax);

    void remove_finer_mesh(BlockID const &);

    void remove_finer_mesh();

    void re_mesh_finer(BlockID const &);

    void sync();


};

template<typename CS>
using finest_mesh = Mesh<CS, tags::multi_block<0> >;


template<typename CS, int LEVEL>
Mesh<CS, tags::multi_block<LEVEL>>::Mesh(point_type const &x_min, point_type const &x_max, index_tuple const &d) :
        m_coord_min_(x_min), m_coord_max_(x_max)
{

}

template<typename CS, int LEVEL>
Mesh<CS, tags::multi_block<LEVEL>>::~Mesh()
{

}

template<typename CS, int LEVEL>
BlockID  Mesh<CS, tags::multi_block<LEVEL>>::add_finer_mesh(index_tuple const &i_min, index_tuple const &i_max)
{

    /**
     * TODO:
     *  1. check overlap
     *  2.
     */
    point_type x_min, x_max;

    auto res = m_finer_mesh_list_.emplace(BlockID(LEVEL), finer_mesh(x_min, x_max, i_max - i_min));


    return res.first->first;
}

template<typename CS, int LEVEL>
void Mesh<CS, tags::multi_block<LEVEL>>::remove_finer_mesh(BlockID const &id)
{
    if (m_finer_mesh_list_.find(id) != m_finer_mesh_list_.end())
    {
        m_finer_mesh_list_[id].remove_finer_mesh();
        m_finer_mesh_list_.erase(id);

        signal_finer(id);
    }
}


template<typename CS, int LEVEL>
void Mesh<CS, tags::multi_block<LEVEL>>::sync()
{
    for (auto &item:m_finer_mesh_list_)
    {
        signal_sync(item.first);
    }
}


}// namespace simpla

#endif //SIMPLA_MESH_BLOCK_BLOCK_H_
