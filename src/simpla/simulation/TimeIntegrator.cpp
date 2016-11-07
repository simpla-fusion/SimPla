//
// Created by salmon on 16-11-7.
//

#include "TimeIntegrator.h"

namespace simpla { namespace simulation
{
struct TimeIntegrator::pimpl_s
{
    std::shared_ptr<mesh::Worker> m_worker_;
    std::shared_ptr<mesh::Atlas> m_atlas_;
    Real m_refine_ratio_ = 2.0;
};

TimeIntegrator::TimeIntegrator(std::string const &s_name)
        : toolbox::Object(s_name)
{

}

TimeIntegrator::~TimeIntegrator()
{

}

void TimeIntegrator::update_level(int l0, int l1)
{

}

void TimeIntegrator::coarsen_level(int l)
{

}

void TimeIntegrator::advance(Real dt, int level)
{
    if (m_pimpl_->m_atlas_->count(level) > 0)
    {
        update_level(level, level + 1);
        for (int i = 0; i < m_pimpl_->m_refine_ratio_; ++i)
        {
            advance(dt / m_pimpl_->m_refine_ratio_, level + 1);
        }
        update_level(level + 1, level);

    }
}

std::string TimeIntegrator::name() const { return toolbox::Object::name(); }

std::ostream &TimeIntegrator::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << " " << name() << " = {"
       << " atlas = {" << m_pimpl_->m_atlas_ << " } ,"
       << " data = {" << m_pimpl_->m_worker_ << "}"
       << "}" << std::endl;
    return os;
}

void TimeIntegrator::load(data::DataBase const &) { UNIMPLEMENTED; }

void TimeIntegrator::save(data::DataBase *) const { UNIMPLEMENTED; }
}}//namespace simpla { namespace simulation
