//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_TRANSITIONMAP_H
#define SIMPLA_TRANSITIONMAP_H

#include <type_traits>
#include "../toolbox/Log.h"
#include "../toolbox/nTuple.h"

#include "MeshCommon.h"
#include "Chart.h"

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
    typedef mesh::Chart Chart;

public:
    TransitionMap(Chart const *m, Chart const *n, index_box_type i_box, index_tuple offset, int flag);

    ~TransitionMap();

    int flag;

//private:

    //TODO use geometric object replace box
    index_box_type m_overlap_idx_box_;
    MeshEntityId m_offset_;
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

    DECL_RET_TYPE ((g(map(x))))

    template<typename Tg, typename Tf>
    void pull_back(Tg const &g, Tf *f, mesh::MeshEntityType entity_type = mesh::VERTEX) const
    {
        first->range(m_overlap_idx_box_, entity_type).foreach(
                [&](mesh::MeshEntityId s)
                {
//                    (*f)[first->hash(s)] =
//                            first->sample(s, pull_back(g, first->point(s)));
                });
    }

    template<typename TFun>
    int direct_map(MeshEntityType entity_type, TFun const &body) const
    {
        first->range(m_overlap_idx_box_, entity_type).foreach(
                [&](mesh::MeshEntityId const &s) { body(s, direct_map(s)); });
    }


    int direct_pull_back(void *f, void const *g, size_type ele_size_in_byte, MeshEntityType entity_type) const;


    template<typename TV>
    int direct_pull_back(TV *f, TV const *g, MeshEntityType entity_type) const
    {
        first->range(m_overlap_idx_box_, entity_type).foreach(
                [&](mesh::MeshEntityId const &s) { f[first->hash(s)] = g[second->hash(direct_map(s))]; });
    }


    template<typename TScalar>
    void push_forward(point_type const &x, TScalar const *v, TScalar *u) const {}


};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_TRANSITIONMAP_H
