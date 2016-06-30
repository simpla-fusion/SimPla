/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <type_traits>
#include "../gtl/Log.h"
#include "../gtl/nTuple.h"
#include "../parallel/DistributedObject.h"

#include "MeshCommon.h"
#include "MeshBase.h"
#include "../io/IOStream.h"

namespace simpla { namespace mesh
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */
typedef mesh::MeshBase Chart;

enum { SP_MB_SYNC = 0x1, SP_MB_COARSEN = 0x2, SP_MB_REFINE = 0x4 };

/**
 *   TransitionMap: \f$\psi\f$,
 *   *Mapping: Two overlapped charts \f$x\in M\f$ and \f$y\in N\f$, and a mapping
 *    \f[
 *       \psi:M\rightarrow N,\quad y=\psi\left(x\right)
 *    \f].
 *   * Pull back: Let \f$g:N\rightarrow\mathbb{R}\f$ is a function on \f$N\f$,
 *     _pull-back_ of function \f$g\left(y\right)\f$ induce a function on \f$M\f$
 *   \f[
 *       \psi^{*}g&\equiv g\circ\psi,\;\psi^{*}g=&g\left(\psi\left(x\right)\right)
 *   \f]
 *
 *
 */
struct TransitionMap
{

public:
    TransitionMap(Chart const *m, Chart const *n, int flag = SP_MB_SYNC);

    ~TransitionMap();

    int flag;

    Chart const *first;
    Chart const *second;


    virtual int map(point_type *) const;

    virtual point_type map(point_type const &) const;

    virtual mesh::MeshEntityId direct_map(mesh::MeshEntityId) const;

    virtual void push_forward(point_type const &x, Real const *v, Real *u) const
    {

        u[0] = v[0];
        u[1] = v[1];
        u[2] = v[2];
    }


    point_type operator()(point_type const &x) const { return map(x); }


    template<typename Tg>
    auto pull_back(Tg const &g, point_type const &x) const
    DECL_RET_TYPE((g(map(x))))

    template<typename Tg, typename Tf>
    void pull_back(Tg const &g, Tf *f, mesh::MeshEntityType entity_type = mesh::VERTEX) const
    {
        first->range(m_overlap_region_M_, entity_type).foreach(
                [&](mesh::MeshEntityId s)
                {
//                    (*f)[first->hash(s)] =
//                            first->sample(s, pull_back(g, first->point(s)));
                });
    }

    template<typename TFun>
    int direct_map(MeshEntityType entity_type, TFun const &fun) const
    {
        parallel::serial_foreach(
                first->range(m_overlap_region_M_, entity_type),
                [&](mesh::MeshEntityId const &s) { fun(s, direct_map(s)); }
        );
    }


    int direct_pull_back(void *f, void const *g, size_type ele_size_in_byte, MeshEntityType entity_type) const;


    template<typename TV>
    int direct_pull_back(TV *f, TV const *g, MeshEntityType entity_type) const
    {
        first->range(m_overlap_region_M_, entity_type).foreach(
                [&](mesh::MeshEntityId const &s) { f[first->hash(s)] = g[second->hash(direct_map(s))]; });
    }


    template<typename TScalar>
    void push_forward(point_type const &x, TScalar const *v, TScalar *u) const
    {

    }


//private:

    //TODO use geometric object replace box
    box_type m_overlap_region_M_;
    mesh::MeshEntityId m_offset_;

};


class Atlas
{
public:

    MeshBlockId add_block(std::shared_ptr<Chart> p_m);

    std::shared_ptr<Chart> get_block(mesh::MeshBlockId m_id) const;

    void remove_block(MeshBlockId const &m_id);

//    std::shared_ptr<MeshBase> extent_block(mesh::MeshBlockId first, int const *offset_direction, size_type width);
//
//    std::shared_ptr<MeshBase> refine_block(mesh::MeshBlockId first, box_type const &);
//
//    std::shared_ptr<MeshBase> coarsen_block(mesh::MeshBlockId first, box_type const &);


    std::map<mesh::MeshBlockId, std::shared_ptr<Chart>> const &at_level(int l = 0) const { return m_; };

    io::IOStream &save(io::IOStream &os) const;

    io::IOStream &load(io::IOStream &is);

    void add_adjacency(mesh::MeshBlockId first, mesh::MeshBlockId second, int flag);

private:
    typedef std::multimap<mesh::MeshBlockId, std::shared_ptr<TransitionMap>> adjacency_list_t;

    adjacency_list_t m_adjacency_list_;

    std::map<mesh::MeshBlockId, std::shared_ptr<Chart>> m_;

public:
    auto get_adjacencies(mesh::MeshBlockId first) const DECL_RET_TYPE((this->m_adjacency_list_.equal_range(first)))


};
}}//namespace simpla{namespace get_mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
