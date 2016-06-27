/**
 * @file context.cpp
 *
 *  Created on: 2015-1-2
 *      Author: salmon
 */

#include "Context.h"
#include "ProblemDomain.h"
//#include <functional>
//#include <iostream>
//#include <map>
//#include <memory>
//#include <string>
//#include <tuple>


namespace simpla { namespace simulation
{

struct Context::pimpl_s
{
    std::map<mesh::MeshBlockId, std::shared_ptr<ProblemDomain>> m_domains_;
    mesh::Atlas m_atlas_;

};

Context::Context() : m_pimpl_(new pimpl_s)
{

};

Context::~Context()
{

};

void
Context::setup()
{

};

void
Context::teardown()
{

};

std::ostream &
Context::print(std::ostream &os, int indent) const
{
    os << "{" << std::endl;
    for (auto const &item:m_pimpl_->m_domains_)
    {
        item.second->print(os, indent + 1);
    }
    os << "}" << std::endl;
    return os;
}


void
Context::add_mesh(std::shared_ptr<mesh::Chart> m, int level)
{
    m_pimpl_->m_atlas_.add_block(m);
}

std::shared_ptr<mesh::Chart>
Context::get_mesh_chart(mesh::MeshBlockId id, int level) const
{
    return m_pimpl_->m_atlas_.get_block(id);
}


std::shared_ptr<ProblemDomain>
Context::get_domain(mesh::MeshBlockId id) const
{
    return m_pimpl_->m_domains_.at(id);
};


std::shared_ptr<ProblemDomain>
Context::add_domain(std::shared_ptr<ProblemDomain> pb, int level)
{
    auto id = pb->m->id();

    m_pimpl_->m_domains_.emplace(std::make_pair(pb->m->id(), pb));
//
//    if (m_pimpl_->m_atlas_.find(id) == m_pimpl_->m_atlas_.end())
//    {
//        add_mesh(const_cast<mesh::Chart *>(pb->m)->shared_from_this(), at_level);
//    }
    return pb;
}


io::IOStream &
Context::check_point(io::IOStream &os) const
{
    for (auto const &item:m_pimpl_->m_domains_)
    {
        item.second->check_point(os);

    }
    return os;
}

io::IOStream &
Context::save(io::IOStream &os) const
{

    for (auto const &item:m_pimpl_->m_domains_)
    {
        item.second->save(os);

    }
    return os;
}

io::IOStream &
Context::load(io::IOStream &is)
{
    for (auto const &item:m_pimpl_->m_domains_)
    {
        item.second->load(is);
    }
    return is;
}

void
Context::run(Real dt, int level)
{

    //TODO async run

#ifdef ENABLE_AMR
    update(level + 1, mesh::SP_MB_REFINE); //  push data to next level
    for (int i = 0; i < m_refine_ratio; ++i)
    {
        run(dt / m_refine_ratio, level + 1);
    }
#endif

    for (auto const &chart_node: m_pimpl_->m_atlas_.at_level(level))
    {
        for (auto p_it = m_pimpl_->m_domains_.find(chart_node.second->id());
             p_it != m_pimpl_->m_domains_.end(); ++p_it)
        {
            p_it->second->next_step(dt);
        };
        chart_node.second->next_step(dt);
    }
    update(level, mesh::SP_MB_COARSEN | mesh::SP_MB_SYNC);

    next_time_step(dt);
};


void
Context::update(int level, int flag)
{
    //TODO async update

    for (auto &mesh_chart: m_pimpl_->m_atlas_.at_level(level))
    {
        auto this_domain = m_pimpl_->m_domains_.find(mesh_chart.second->id());
        if (this_domain != m_pimpl_->m_domains_.find(mesh_chart.second->id()))
        {
            auto r = m_pimpl_->m_atlas_.get_adjacencies(mesh_chart.first);
            for (auto it = std::get<0>(r), ie = std::get<1>(r); it != ie; ++it)
            {
                auto other_domain = m_pimpl_->m_domains_.find(it->second->second->id());
                if ((it->second->flag & flag) != 0x0 && other_domain != m_pimpl_->m_domains_.end())
                {
                    this_domain->second->sync(*(it->second), *(other_domain->second));
                }

            };
        };
    }

};


}} // namespace simpla { namespace simulation


