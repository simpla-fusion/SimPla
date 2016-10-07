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
#include "Chart.h"
#include "../toolbox/IOStream.h"

namespace simpla { namespace mesh
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */
typedef mesh::Chart Chart;

struct TransitionMap;

class Atlas
{
public:

    MeshBlockId add_block(std::shared_ptr<Chart> p_m);

    std::shared_ptr<Chart> get_block(mesh::MeshBlockId m_id) const;

    void remove_block(MeshBlockId const &m_id);

//    std::shared_ptr<Chart> extent_block(mesh::MeshBlockId first, int const *offset_direction, size_type width);
//
//    std::shared_ptr<Chart> refine_block(mesh::MeshBlockId first, box_type const &);
//
//    std::shared_ptr<Chart> coarsen_block(mesh::MeshBlockId first, box_type const &);


    std::map<mesh::MeshBlockId, std::shared_ptr<Chart>> const &at_level(int l = 0) const { return m_nodes_; };

    toolbox::IOStream &save(toolbox::IOStream &os) const;

    toolbox::IOStream &load(toolbox::IOStream &is);

    void add_adjacency(std::shared_ptr<TransitionMap>);

    std::shared_ptr<TransitionMap> add_adjacency(const Chart *first, const Chart *second, int flag);

    std::shared_ptr<TransitionMap> add_adjacency(MeshBlockId first, MeshBlockId second, int flag);

    void add_adjacency2(const Chart *first, const Chart *second, int flag);

//#ifndef NDEBUG
//    private:
//#endif
    typedef std::multimap<mesh::MeshBlockId, std::shared_ptr<TransitionMap>> adjacency_list_t;

    adjacency_list_t m_adjacency_list_;

    std::map<mesh::MeshBlockId, std::shared_ptr<Chart>> m_nodes_;

    std::multimap<mesh::MeshBlockId, std::shared_ptr<TransitionMap>> m_out_edge_;
    std::multimap<mesh::MeshBlockId, std::shared_ptr<TransitionMap>> m_in_edge_;

public:
    std::pair<typename adjacency_list_t::const_iterator, typename adjacency_list_t::const_iterator>
    get_adjacencies(mesh::MeshBlockId first) const { return m_adjacency_list_.equal_range(first); }


};
}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
