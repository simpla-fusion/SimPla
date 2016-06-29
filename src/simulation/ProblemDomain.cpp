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
    std::map<std::string, mesh::MeshAttribute *> m_attr_;
    parallel::DistributedObject m_dist_obj_;
};

ProblemDomain::ProblemDomain() : m_mesh_(nullptr), m_pimpl_(new pimpl_s) { }

ProblemDomain::ProblemDomain(const mesh::MeshBase *msh) : m_mesh_(msh), m_pimpl_(new pimpl_s) { };

ProblemDomain::~ProblemDomain() { teardown(); }


const mesh::MeshAttribute *
ProblemDomain::attribute(std::string const &s_name) const
{
    return m_pimpl_->m_attr_.at(s_name);
};

void ProblemDomain::add_attribute(mesh::MeshAttribute *attr, std::string const &s_name)
{

    m_pimpl_->m_attr_.emplace(std::make_pair(s_name, attr));

};

void ProblemDomain::deploy()
{
    LOGGER << "deploy problem domain [" << get_class_name() << "]" << std::endl;
};

void ProblemDomain::teardown()
{
    if (m_mesh_ != nullptr)
    {
        m_mesh_ = nullptr;
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
    m_mesh_->print(os, indent + 2);
    os << std::setw(indent + 1) << " }," << std::endl;

//    os << std::setw(indent + 1) << " time =" << m_self_->m_time_ << ", dt =" << m_self_->m_dt_ << "," << std::endl;


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
    if (!m_properties_["DISABLE_LOAD"])
    {
        UNIMPLEMENTED;
    }
    return is;
}

io::IOStream &
ProblemDomain::save(io::IOStream &os, int flag) const
{
    if (!m_properties_["DISABLE_SAVE"])
    {
        auto pwd = os.pwd();

        for (auto const &item:m_pimpl_->m_attr_)
        {
            if (!item.second->empty())
            {
                os.open(item.first + "/");
                os.write(m_mesh_->name(), item.second->dataset(), flag);
                os.open(pwd);
            }
        }
    }
    return os;
};


void ProblemDomain::sync(mesh::TransitionMap const &t_map, ProblemDomain const &other)
{
//    for (auto const &item:m_pimpl_->m_attr_)
//    {
//        if (!item.second->empty())
//        {
//            t_map.direct_pull_back(item.second->data().get(), other.attribute(item.first)->data().get(),
//                                   item.second->entity_size_in_byte(), item.second->entity_type());
//        }
//
//    }
}

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
