/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "ProblemDomain.h"
#include "../io/IOStream.h"

namespace simpla { namespace simulation
{
struct ProblemDomain::pimpl_s
{
//    Real m_dt_ = 0;
//    Real m_time_ = 0;
    std::map<std::string, std::shared_ptr<mesh::MeshAttribute> > m_attr_;
    parallel::DistributedObject m_dist_obj_;
};

ProblemDomain::ProblemDomain() : m(nullptr), m_pimpl_(new pimpl_s) { }

ProblemDomain::ProblemDomain(const mesh::MeshBase *msh) : m(msh), m_pimpl_(new pimpl_s) { };

ProblemDomain::~ProblemDomain() { teardown(); }


//Real const &ProblemDomain::dt() const { return m_pimpl_->m_dt_; }
//
//void ProblemDomain::dt(Real pdt) { m_pimpl_->m_dt_ = pdt; }
//
//Real ProblemDomain::time() const { return m_pimpl_->m_time_; }
//
//void ProblemDomain::time(Real t) { m_pimpl_->m_time_ = t; }

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

void ProblemDomain::setup(ConfigParser const &dict)
{

    init(dict);
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


//void
//ProblemDomain::run(Real stop_time, int num_of_step)
//{
//    Real inc_t = (num_of_step > 0) ? ((stop_time - time()) / num_of_step) : dt();
//
//    while (stop_time - time() > inc_t)
//    {
//        next_time_step(inc_t);
//
////        sync();
//
//        time(time() + inc_t);
//    }
//    next_time_step(stop_time - time());
//    sync();
//    time(stop_time);
//}


std::ostream &
ProblemDomain::print(std::ostream &os, int indent) const
{
    auto it = m_pimpl_->m_attr_.begin();
    auto ie = m_pimpl_->m_attr_.end();

    os << std::setw(indent) << " ProblemDomain={" << std::endl;

    os << std::setw(indent + 1) << " Type=\"" << get_class_name() << "\"," << std::endl;

    os << std::setw(indent + 1) << " Mesh= {" << std::endl;
    m->print(os, indent + 2);
    os << std::setw(indent + 1) << " }," << std::endl;

//    os << std::setw(indent + 1) << " time =" << m_pimpl_->m_time_ << ", dt =" << m_pimpl_->m_dt_ << "," << std::endl;


    os << std::setw(indent + 1) << " Attributes= {  ";


    for (auto const &item:m_pimpl_->m_attr_)
    {
        os << " " << item.first;
//        os << "={";
//        item.second->print(os, indent + 2);
//        os << "}";
        os << ",";
    }

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
    auto m_id = m->uuid();
    for (auto const &item:m_pimpl_->m_attr_)
    {
        os.write(item.first, item.second->dataset(m_id), io::SP_NEW);
    }
    return os;
}

io::IOStream &
ProblemDomain::check_point(io::IOStream &os) const
{
    auto m_id = m->uuid();
    for (auto const &item:m_pimpl_->m_attr_)
    {
        auto ds = item.second->dataset(m_id);
        if (ds.is_valid()) { os.write(item.first, ds, io::SP_RECORD); }

    }
    return os;
}


//void ProblemDomain::view(get_mesh::MeshBlockId const &) { }
//
//void ProblemDomain::update_ghost_from(get_mesh::MeshBase const &other) { };

bool ProblemDomain::same_as(mesh::MeshBase const &) const
{
    UNIMPLEMENTED;
    return false;
};

//std::vector<mesh::box_type> ProblemDomain::refine_boxes() const
//{
//    UNIMPLEMENTED;
//    return std::vector<mesh::box_type>();
//}
//
//void ProblemDomain::refine(mesh::MeshBase const &other) { UNIMPLEMENTED; };
//
//bool ProblemDomain::coarsen(mesh::MeshBase const &other)
//{
//    UNIMPLEMENTED;
//    return false;
//};


}} //namespace simpla { namespace simulation
