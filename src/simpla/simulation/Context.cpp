/**
 * @file context.cpp
 *
 *  Created on: 2015-1-2
 *      Author: salmon
 */

#include "Context.h"
#include <simpla/mesh/DomainBase.h>
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
    std::map<uuid, std::shared_ptr<mesh::DomainBase>> m_domains_;
    mesh::Atlas m_atlas_;

};

Context::Context() : m_pimpl_(new pimpl_s) {};

Context::~Context() {};

void Context::setup() {};

void Context::teardown() {};

std::ostream &
Context::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << " " << " DomainBase = {" << std::endl;
    for (auto const &item:m_pimpl_->m_domains_)
    {
        os << std::setw(indent + 1) << " " << "{" << std::endl;

        item.second->print(os, indent + 2);

        os << std::setw(indent + 1) << " " << "}," << std::endl;
    }
    os << std::setw(indent) << " " << " }," << std::endl;
    return os;
}

//mesh_as::Atlas &
//Context::atlas() { return m_pimpl_->m_atlas_; };
//
//mesh_as::Atlas const &
//Context::get_mesh_atlas() const { return m_pimpl_->m_atlas_; };
//
//mesh_as::MeshBlockId
//Context::add_mesh(std::shared_ptr<mesh_as::MeshBase> m) { return m_pimpl_->m_atlas_.add_block(m); }
//
//std::shared_ptr<const mesh_as::MeshBase>
//Context::get_mesh_block(mesh_as::MeshBlockId id) const { return m_pimpl_->m_atlas_.get_block(id); }
//
//std::shared_ptr<mesh_as::MeshBase>
//Context::get_mesh_block(mesh_as::MeshBlockId id) { return m_pimpl_->m_atlas_.get_block(id); }

std::shared_ptr<mesh::DomainBase>
Context::get_domain(uuid id) const { return m_pimpl_->m_domains_.at(id); };


std::shared_ptr<mesh::DomainBase>
Context::add_domain(std::shared_ptr<mesh::DomainBase> pb)
{
    assert(pb != nullptr);
    auto it = m_pimpl_->m_domains_.find(pb->id());
    if (it == m_pimpl_->m_domains_.end())
    {
        m_pimpl_->m_domains_.emplace(std::make_pair(pb->id(), pb));
    }
    // else
//    {
//        std::shared_ptr<mesh::DomainBase> *p = &(m_pimpl_->m_nodes_[pb->mesh_as()->id()]);
//        while (*p != nullptr) { p = &((*p)->next()); }
//        *p = pb;
//    }
    return pb;
}

//toolbox::IOStream &
//Context::save_mesh(toolbox::IOStream &os) const
//{
////    m_pimpl_->m_atlas_.save(os);
//    return os;
//}
//
//
//toolbox::IOStream &
//Context::load_mesh(toolbox::IOStream &is)
//{
//    UNIMPLEMENTED;
//    return is;
//}

toolbox::IOStream &
Context::save(toolbox::IOStream &os, int flag) const
{
//    for (auto const &item:m_pimpl_->m_domains_) { item.second->save(os, flag); }
    return os;
}

toolbox::IOStream &
Context::check_point(toolbox::IOStream &os) const { return save(os, toolbox::SP_RECORD); }

toolbox::IOStream &
Context::load(toolbox::IOStream &is)
{
    UNIMPLEMENTED;

    return is;
}

void
Context::run(Real dt, int level)
{

    //TODO async run

//#ifdef ENABLE_AMR
//    sync(level + 1, mesh_as::SP_MB_REFINE); //  push data to next level
//    for (int i = 0; i < m_refine_ratio; ++i)
//    {
//        run(dt / m_refine_ratio, level + 1);
//    }
//#endif

//    for (auto const &chart_node: m_pimpl_->m_atlas_.level(level))
//    {
//        auto p_it = m_pimpl_->m_domains_.find(chart_node.second->id());
//
//        if (p_it != m_pimpl_->m_domains_.end())
//        {
//
//            auto p = p_it->second;
//            while (p != nullptr)
//            {
//                p->next_step(dt);
//                p = p->next();
//            }
//        };
//
////        chart_node.second->next_step(dt);
//    }
    next_time_step(dt);
};


void
Context::sync(int level, int flag)
{
    //TODO async sync
//    for (auto const &mesh_chart: m_pimpl_->m_atlas_.at_level(level))
//    {
//        auto this_domain = m_pimpl_->m_domains_.find(mesh_chart.second->id());
//        if (this_domain != m_pimpl_->m_domains_.end())
//        {
//            auto r = m_pimpl_->m_atlas_.get_adjacencies(mesh_chart.first);
//            for (auto it = std::get<0>(r), ie = std::get<1>(r); it != ie; ++it)
//            {
////                auto other_domain = m_pimpl_->m_nodes_.find(it->second->second->id());
////                if (other_domain != m_pimpl_->m_nodes_.end() && (it->second->flag & flag != 0))
////                {
////                    this_domain->second->sync(*(it->second), *(other_domain->second));
////                }
//            };
//        };
//    }

};


}} // namespace simpla { namespace simulation


