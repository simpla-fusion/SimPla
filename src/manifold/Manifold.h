//
// Created by salmon on 16-6-24.
//

#ifndef SIMPLA_MANIFOLD_H
#define SIMPLA_MANIFOLD_H

#include <memory>
#include "../mesh/MeshCommon.h"
#include "../mesh/MeshEntity.h"
#include "../gtl/type_traits.h"

namespace simpla { namespace manifold
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */
struct ChartBase
{
    virtual mesh::point_type point(mesh::MeshEntityId const &s) const = 0;

    virtual size_type hash(mesh::MeshEntityId s) const = 0;

    virtual mesh::MeshEntityRange range(mesh::box_type const &b, int entity_type) const = 0;

    virtual mesh::MeshBlockId id() const = 0;

    template<typename TFrom>
    Real sample(mesh::MeshEntityId s, TFrom const &v) const { return 0.0; }

    Real sample(mesh::MeshEntityId s, Real const &v) const { return v; }


};

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
    TransitionMap(ChartBase const *m, ChartBase const *n);

    ~TransitionMap();


    virtual int map(mesh::point_type *) const = 0;

    virtual mesh::point_type map(mesh::point_type const &) const = 0;

    virtual mesh::MeshEntityId direct_map(mesh::MeshEntityId) const = 0;

    virtual void push_forward(mesh::point_type const &x, Real const *v, Real *u) const
    {

        u[0] = v[0];
        u[1] = v[1];
        u[2] = v[2];
    }


    mesh::point_type operator()(mesh::point_type const &x) const { return map(x); }


    template<typename Tg>
    auto pull_back(Tg const &g, mesh::point_type const &x) const
    DECL_RET_TYPE((g(map(x))))

    template<typename Tg, typename Tf>
    void pull_back(Tg const &g, Tf *f, int entity_type = mesh::VERTEX) const
    {
        m_chart_M_->range(m_overlap_region_M_, entity_type).foreach(
                [&](mesh::MeshEntityId s)
                {
                    (*f)[m_chart_M_->hash(s)] =
                            m_chart_M_->sample(s, pull_back(g, m_chart_M_->point(s)));
                });
    }


    int direct_pull_back(Real const *g, Real *f, int entity_type = mesh::VERTEX) const;


    template<typename TScalar>
    void push_forward(mesh::point_type const &x, TScalar const *v, TScalar *u) const
    {

    }

private:
    ChartBase const *m_chart_M_;
    ChartBase const *m_chart_N_;
    mesh::box_type m_overlap_region_M_;
};


class Atlas
{
public:
    void add_chart(std::string const &, std::shared_ptr<ChartBase>);

    void add_map(std::string const &, std::string const &, std::shared_ptr<TransitionMap>);

    void update(mesh::MeshBlockId m_id, mesh::MeshBlockId n_id, Real const *l, Real *r, int entity_type) const;


    ChartBase const *get_chart(mesh::MeshBlockId m_id) const;

    ChartBase *get_chart(mesh::MeshBlockId m_id);

    TransitionMap const &get_map(mesh::MeshBlockId m_id, mesh::MeshBlockId n_id) const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}}//namespace simpla { namespace manifold

#endif //SIMPLA_MANIFOLD_H
