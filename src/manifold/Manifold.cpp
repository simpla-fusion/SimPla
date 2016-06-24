//
// Created by salmon on 16-6-24.
//
#include "Manifold.h"

namespace simpla { namespace manifold
{

TransitionMap::TransitionMap(ChartBase const *m, ChartBase const *n)
        : m_chart_M_(m), m_chart_N_(n) { }

TransitionMap::~TransitionMap() { }

int TransitionMap::direct_pull_back(Real const *g, Real *f, int entity_type) const
{
    m_chart_M_->range(m_overlap_region_M_, entity_type).foreach(
            [&](mesh::MeshEntityId const &s)
            {
                f[m_chart_M_->hash(s)] = g[m_chart_N_->hash(direct_map(s))];
            });
};

struct Atlas::pimpl_s
{
    std::map<mesh::MeshBlockId, std::map<mesh::MeshBlockId, std::shared_ptr<TransitionMap>>> m_adjacency_matrix_;
    std::multimap<int, TransitionMap> m_edges_;
};

void Atlas::add_chart(std::string const &, std::shared_ptr<ChartBase>) { }


void Atlas::add_map(std::string const &, std::string const &, std::shared_ptr<TransitionMap>)
{

}

void Atlas::update(mesh::MeshBlockId m_id, mesh::MeshBlockId n_id, Real const *g, Real *f, int entity_type) const
{
    m_pimpl_->m_adjacency_matrix_.at(m_id).at(n_id)->direct_pull_back(g, f, entity_type);
}

}}//namespace simpla { namespace manifold
