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
#include "MeshBase.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{


class MeshAttributeVisitorBase;

template<typename T> class MeshAttributeVisitor;

template<typename ...> struct TransitionMap;

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
struct TransitionMapBase
{
    std::shared_ptr<MeshBase> m_dst_, m_src_;

    TransitionMapBase() {};

    TransitionMapBase(MeshBase const &, MeshBase const &) {};

    virtual  ~TransitionMapBase() {};

    virtual point_type map(point_type const &x) const { return x; }

    point_type operator()(point_type const &x) const { return map(x); }

//    virtual void push_forward(AttributeBase const &src, AttributeBase *dest) const =0;
//
//    virtual void pull_back(AttributeBase const &src, AttributeBase *dest) const =0;


};


template<typename M, typename N>
struct TransitionMap<M, N> :
        public TransitionMapBase
{
    MeshBase m_overlap_;
    EntityRange m_range0_;

    virtual void push_forward(AttributeBase const &src, AttributeBase *dest) const
    {


    };

    template<size_t I, typename TF, typename TG>
    void push_forward_impl(index_const<I>, TF const &f, TG *g) const
    {
//        m_overlap_.range(I).foreach([&](MeshEntityId const &s) { (*g)[s] = f[s]; });
    };



    template<typename TDest, typename TSrc>
    struct Visitor : public MeshAttributeVisitor<TSrc>
    {

    };

    virtual void pull_back(AttributeBase const &src, AttributeBase *dest) const
    {

    };

};


template<typename TM>
std::shared_ptr<TransitionMapBase>
createTransitionMap(std::shared_ptr<TM> const &src, std::shared_ptr<TM> const &dest)
{

}
}}//namespace simpla { namespace mesh

#endif //SIMPLA_TRANSITIONMAP_H
