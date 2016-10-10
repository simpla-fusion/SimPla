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
#include "Block.h"

namespace simpla { namespace mesh
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */

struct TransitionMap;

class Atlas
{
public:

    Block::id_type add_block(std::shared_ptr<Block> p_m);

    std::shared_ptr<Block> get_block(Block::id_type m_id) const;

    void remove_block(Block::id_type const &m_id);

    std::map<Block::id_type, std::shared_ptr<Block>> const &at_level(int l = 0) const { return m_nodes_; };


    void add_adjacency(std::shared_ptr<TransitionMap>);

    std::shared_ptr<TransitionMap>
    add_adjacency(std::shared_ptr<const Block> first, std::shared_ptr<const Block> second);

    std::shared_ptr<TransitionMap> add_adjacency(MeshBlockId first, MeshBlockId second);

    void add_adjacency2(std::shared_ptr<const Block> first, std::shared_ptr<const Block> second);

//#ifndef NDEBUG
//    private:
//#endif
    typedef std::multimap<Block::id_type, std::shared_ptr<TransitionMap>> adjacency_list_t;

    adjacency_list_t m_adjacency_list_;

    std::map<Block::id_type, std::shared_ptr<Block>> m_nodes_;

    std::multimap<Block::id_type, std::shared_ptr<TransitionMap>> m_out_edge_;
    std::multimap<Block::id_type, std::shared_ptr<TransitionMap>> m_in_edge_;

public:
    std::pair<typename adjacency_list_t::const_iterator, typename adjacency_list_t::const_iterator>
    get_adjacencies(Block::id_type first) const { return m_adjacency_list_.equal_range(first); }


};
}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
