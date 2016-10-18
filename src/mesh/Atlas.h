/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <type_traits>
#include "../toolbox/Log.h"
#include "../toolbox/nTuple.h"
#include "MeshCommon.h"
#include "MeshBase.h"
#include "TransitionMap.h"

namespace simpla { namespace mesh
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */


class Atlas
{
public:

    MeshBase::id_type add_block(std::shared_ptr<MeshBase> p_m);

    std::shared_ptr<MeshBase> get_block(MeshBase::id_type m_id) const;

    void remove_block(MeshBase::id_type const &m_id);

    std::map<MeshBase::id_type, std::shared_ptr<MeshBase>> const &at_level(int l = 0) const { return m_nodes_; };


    std::shared_ptr<TransitionMapBase> add_adjacency(std::shared_ptr<TransitionMapBase>);
//
//    std::shared_ptr<TransitionMapBase>
//    add_adjacency(std::shared_ptr<const MeshBase> first, std::shared_ptr<const MeshBase> second);
//
//    std::shared_ptr<TransitionMapBase>
//    add_adjacency(MeshBlockId first, MeshBlockId second);
//
//    std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
//    add_connection(std::shared_ptr<const MeshBase> first, std::shared_ptr<const MeshBase> second);


    template<typename TM, typename TN>
    std::shared_ptr<TransitionMap<TM, TN> >
    add_adjacency(std::shared_ptr<TM> const &first, std::shared_ptr<TN> const &second)
    {
        auto res = std::make_shared<TransitionMap<TM, TN>>(first, second);
        add_adjacency(std::dynamic_pointer_cast<TransitionMapBase>(res));
        return res;
    }

    template<typename TM, typename TN>
    std::tuple<std::shared_ptr<TransitionMap<TM, TN> >, std::shared_ptr<TransitionMap<TN, TM> > >
    add_connection(std::shared_ptr<TM> const &first, std::shared_ptr<TN> const &second)
    {
        return std::make_tuple(add_adjacency(first, second), add_adjacency(second, first));

    };

//#ifndef NDEBUG
//    private:
//#endif
    typedef std::multimap<MeshBase::id_type, std::shared_ptr<TransitionMapBase>> adjacency_list_t;
//
//    adjacency_list_t m_adjacency_list_;

    std::map<MeshBase::id_type, std::shared_ptr<MeshBase>> m_nodes_;

    std::multimap<MeshBase::id_type, std::shared_ptr<TransitionMapBase>> m_out_edge_;
    std::multimap<MeshBase::id_type, std::shared_ptr<TransitionMapBase>> m_in_edge_;

public:
    std::pair<typename adjacency_list_t::const_iterator, typename adjacency_list_t::const_iterator>
    get_adjacencies(MeshBase::id_type first) const { return m_out_edge_.equal_range(first); }


};
}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
