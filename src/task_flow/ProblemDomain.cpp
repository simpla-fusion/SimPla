/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "ProblemDomain.h"
#include "../mesh/MeshAtlas.h"
#include "../mesh/MeshAttribute.h"

#include "../io/IOStream.h"

namespace simpla
{
struct ProblemDomain::pimpl_s
{
    Real m_dt_ = 0;
    Real m_time_ = 0;
    std::map<std::string, std::shared_ptr<mesh::MeshAttribute> > m_attr_;
    std::map<int, parallel::DistributedObject> m_dist_obj_;
};

ProblemDomain::ProblemDomain() : m(nullptr), m_pimpl_(new pimpl_s) { }

ProblemDomain::ProblemDomain(std::shared_ptr<const mesh::MeshBase> msh) : m(msh), m_pimpl_(new pimpl_s) { };

ProblemDomain::~ProblemDomain() { teardown(); }


Real const &ProblemDomain::dt() const { return m_pimpl_->m_dt_; }

void ProblemDomain::dt(Real pdt) { m_pimpl_->m_dt_ = pdt; }

Real ProblemDomain::time() const { return m_pimpl_->m_time_; }

void ProblemDomain::time(Real t) { m_pimpl_->m_time_ = t; }

std::shared_ptr<mesh::MeshAttribute>
ProblemDomain::attribute(std::string const &s_name)
{
    if (m_pimpl_->m_attr_.find(s_name) == m_pimpl_->m_attr_.end())
    {
        m_pimpl_->m_attr_.emplace(std::make_pair(s_name, std::make_shared<mesh::MeshAttribute>()));
    }
    return m_pimpl_->m_attr_[s_name];

};

std::shared_ptr<mesh::MeshAttribute const>
ProblemDomain::attribute(std::string const &s_name) const
{
    return m_pimpl_->m_attr_.at(s_name);
};

void ProblemDomain::setup()
{
    auto mesh_block_id = m->uuid();
    auto id = m->short_id();
    auto &dist_obj = m_pimpl_->m_dist_obj_[id];

    for (auto &item:m_pimpl_->m_attr_)
    {
        if (item.second->has(mesh_block_id))
        {
            auto ds = item.second->get_dataset(mesh_block_id);
            dist_obj.add(item.second->short_id(), ds);
        }
    }
    LOGGER << "Setup problem domain [" << get_class_name() << "]" << std::endl;

};

void ProblemDomain::teardown()
{
    if (m != nullptr)
    {
        m = nullptr;
        LOGGER << "Teardown problem domain [" << get_class_name() << "]" << std::endl;
    }
};

std::shared_ptr<ProblemDomain>
ProblemDomain::clone(mesh::MeshBase const &) const
{
    UNIMPLEMENTED;
    return std::shared_ptr<ProblemDomain>(nullptr);
};


void
ProblemDomain::sync()
{
    m_pimpl_->m_dist_obj_[m->short_id()].sync();
}

void
ProblemDomain::run(Real stop_time, int num_of_step)
{
    Real inc_t = dt();

    if (num_of_step > 0) { inc_t = (stop_time - time()) / num_of_step; }

    while (stop_time - time() > inc_t)
    {
        next_step(inc_t);

        sync();

        time(time() + inc_t);
    }
    next_step(stop_time - time());
    sync();
    time(stop_time);
}


std::ostream &
ProblemDomain::print(std::ostream &os, int indent) const
{
    auto it = m_pimpl_->m_attr_.begin();
    auto ie = m_pimpl_->m_attr_.end();

    os << std::setw(indent) << " ProblemDomain={" << std::endl;

    os << std::setw(indent + 1) << " Type=\"" << get_class_name() << "\"," << std::endl;

    os << std::setw(indent + 1) << " Attributes= { \"" << it->first << "\"";

    ++it;

    for (; it != ie; ++it) { os << " , \"" << it->first << "\""; }

    os << "}," << std::endl;

    os << std::setw(indent) << "}," << std::endl;

    return os;
}


io::IOStream &
ProblemDomain::load(io::IOStream &is) const
{
    UNIMPLEMENTED;
    return is;
}


io::IOStream &
ProblemDomain::save(io::IOStream &os) const
{
    for (auto const &item:m_pimpl_->m_attr_)
    {
//        os.write(item.first, item.second->data_set());
    }
    return os;
}

io::IOStream &
ProblemDomain::check_point(io::IOStream &os) const
{
    return os;
}


//void ProblemDomain::view(mesh::MeshBlockId const &) { }
//
//void ProblemDomain::update_ghost_from(mesh::MeshBase const &other) { };

bool ProblemDomain::same_as(mesh::MeshBase const &) const
{
    UNIMPLEMENTED;
    return false;
};

std::vector<mesh::box_type> ProblemDomain::refine_boxes() const
{
    UNIMPLEMENTED;
    return std::vector<mesh::box_type>();
}

void ProblemDomain::refine(mesh::MeshBase const &other) { UNIMPLEMENTED; };

bool ProblemDomain::coarsen(mesh::MeshBase const &other)
{
    UNIMPLEMENTED;
    return false;
};


} //namespace simpla