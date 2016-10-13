//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_TRANSITIONMAP_H
#define SIMPLA_TRANSITIONMAP_H

#include <type_traits>
#include "../toolbox/Log.h"
#include "../toolbox/nTuple.h"
#include "../toolbox/nTupleExt.h"
#include "../toolbox/PrettyStream.h"
#include "MeshCommon.h"
#include "Block.h"

namespace simpla { namespace mesh
{


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
    TransitionMap(Block const &m, Block const &n)
    {
        assert(m.space_id() == m.space_id());


        m_dst_ = m.clone();
        m_dst_->intersection_outer(n.index_box());
        m_dst_->deploy();
        m_src_ = n.clone();
        m_src_->intersection(m.outer_index_box());
        m_src_->deploy();


    };

    virtual  ~TransitionMap() {};

    std::shared_ptr<Block> m_dst_, m_src_;


    virtual point_type map(point_type const &x) const { return x; }

//    virtual mesh::MeshEntityId direct_map(mesh::MeshEntityId) const;

    virtual void push_forward(point_type const &x, Real const *v, Real *u) const
    {
        u[0] = v[0];
        u[1] = v[1];
        u[2] = v[2];
    }


    point_type operator()(point_type const &x) const { return map(x); }


    template<typename Tg>
    auto pull_back(Tg const &g, point_type const &x) const DECL_RET_TYPE ((g(map(x))))

//    template<typename Tg, typename Tf>
//    void pull_back(Tg const &g, Tf *f, mesh::MeshEntityType entity_type = mesh::VERTEX) const
//    {
//        m_dst_->foreach(entity_type,
//                          [&](mesh::MeshEntityId s)
//                          {
//                              (*f)[m_dst_->hash(s)] = m_dst_->sample(s, g(m_src_->point(s)));
//                          });
//    }

    int direct_map(MeshEntityType entity_type,
                   std::function<void(mesh::MeshEntityId const &, mesh::MeshEntityId const &)> const &body) const
    {
        if (m_dst_->size() > 0) m_dst_->foreach(entity_type, [&](mesh::MeshEntityId const &s) { body(s, s); });
    }


    template<typename TV>
    int pointwise_copy(TV *f, TV const *g, MeshEntityType entity_type) const
    {
        if (m_dst_->size() > 0)
            m_dst_->foreach(entity_type, [&](mesh::MeshEntityId const &s) { f[m_dst_->hash(s)] = g[m_src_->hash(s)]; });
    }


    template<typename TScalar>
    void push_forward(point_type const &x, TScalar const *v, TScalar *u) const {}


};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_TRANSITIONMAP_H
