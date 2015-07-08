//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_MESH_AMR_H
#define SIMPLA_MESH_AMR_H

#include <map>
#include <list>
#include <utility>

#include "../mesh_traits.h"
#include "../../geometry/coordinate_system.h"
#include "../../gtl/design_pattern/observer.h"

#include "block.h"
#include "block_id.h"

namespace simpla {
namespace tags {
template<int LEVEL> struct multi_block : public std::integral_constant<int, LEVEL> { };
}
template<typename ...> struct DOFHolder;


template<typename CS, int LEVEL>
struct Mesh<CS, tags::multi_block<LEVEL>> : public Block<geometry::traits::dimension<CS>::value, LEVEL>,
                                            public Observable
{

    typedef Mesh<CS, tags::multi_block<LEVEL>> this_type;

    typedef Mesh<CS, tags::multi_block<LEVEL - 1> > finer_mesh;

    typedef Mesh<CS, tags::multi_block<LEVEL + 1> > coarser_mesh;

    typedef Block<geometry::traits::dimension<CS>::value, LEVEL> topology_type;

    using topology_type::index_tuple;

    typedef traits::point_type_t<CS> point_type;


    enum TRANSFORM { ROTATE0, ROTATE90, ROTATE180, ROTATE270 };

    struct Connection
    {
        index_tuple min, max;

        index_tuple dest_min;

        int transform;
    };

private:

    typedef std::map<BlockID, finer_mesh> mesh_map;

    mesh_map m_finer_mesh_list_;

    std::map<BlockID, std::map<BlockID, Connection> > m_connections_;

    point_type m_coord_min_, m_coord_max_;


public:

    Mesh(point_type const &x_min, point_type const &x_max, index_tuple const &d) :
            topology_type(d), m_coord_min_(x_min), m_coord_max_(x_max)
    {

    }

    ~Mesh()
    {

    }

    template<typename T>
    std::shared_ptr<T> create()
    {
        auto res = std::make_shared<T>(*this);

        Observable::regist(std::dynamic_pointer_cast<Observer>(res));

        return res;
    }

    BlockID add_finer_mesh(index_tuple const &imin, index_tuple const &imax);

    void remove_finer_mesh(BlockID const &);

    void remove_finer_mesh();

    void re_mesh_finer(BlockID const &);

    void sync();

    std::tuple<point_type, point_type> box() const
    { return std::make_tuple(m_coord_min_, m_coord_max_); }

    using topology_type::dimensions;
    using topology_type::hash;


};

template<typename CS>
using finest_mesh = Mesh<CS, tags::multi_block<0> >;


template<typename CS, int LEVEL>
Mesh<CS, tags::multi_block<LEVEL>>::Mesh(point_type const &x_min, point_type const &x_max, index_tuple const &d) :
        topology_type(d), m_coord_min_(x_min), m_coord_max_(x_max)
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

    for (auto &item:m_observers_)
    {
        item->initialize(res.first->first);
    }
    return res.first->first;
}

template<typename CS, int LEVEL>
void Mesh<CS, tags::multi_block<LEVEL>>::remove_finer_mesh(BlockID const &id)
{
    if (m_finer_mesh_list_.find(id) != m_finer_mesh_list_.end())
    {
        m_finer_mesh_list_[id].remove_finer_mesh();
        m_finer_mesh_list_.erase(id);

        for (auto &item:m_observers_)
        {
            item->destroy(id);
        }
    }
}

template<typename CS, int LEVEL>
void Mesh<CS, tags::multi_block<LEVEL>>::remove_finer_mesh()
{
    for (auto &item:m_finer_mesh_list_)
    {
        for (auto &ob:m_observers_)
        {
            ob->destroy(item.first);
        }
    }
    m_finer_mesh_list_.clear();
}

template<typename CS, int LEVEL>
void Mesh<CS, tags::multi_block<LEVEL>>::sync()
{
    for (auto &item:m_finer_mesh_list_)
    {
        for (auto &ob:m_observers_)
        {
            ob->sync(item.first);
        }
    }
}
}//namespace simpla
#endif //SIMPLA_MESH_AMR_H
