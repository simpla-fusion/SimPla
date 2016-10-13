/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "PhysicalDomain.h"
#include "../toolbox/IOStream.h"

namespace simpla { namespace simulation
{
struct PhysicalDomain::pimpl_s
{
//    Real m_dt_ = 0;
//    Real m_time_ = 0;
    std::map<std::string, mesh::Attribute *> m_attr_;
//    parallel::DistributedObject m_dist_obj_;
};

PhysicalDomain::PhysicalDomain() : m_mesh_(nullptr), m_next_(nullptr), m_pimpl_(new pimpl_s) {}

PhysicalDomain::PhysicalDomain(std::shared_ptr<const mesh::Block> msh) : m_mesh_(msh), m_next_(nullptr),
                                                                         m_pimpl_(new pimpl_s) {};

PhysicalDomain::~PhysicalDomain() { teardown(); }


const mesh::Attribute *
PhysicalDomain::attribute(std::string const &s_name) const
{
    return m_pimpl_->m_attr_.at(s_name);
};

void PhysicalDomain::add_attribute(mesh::Attribute *attr, std::string const &s_name)
{
    m_pimpl_->m_attr_.emplace(std::make_pair(s_name, attr));
};

void PhysicalDomain::deploy() { LOGGER << "deploy problem domain [" << get_class_name() << "]" << std::endl; };

void PhysicalDomain::teardown()
{
    if (m_mesh_ != nullptr)
    {
        m_mesh_ = nullptr;
        LOGGER << "Teardown problem domain [" << get_class_name() << "]" << std::endl;
    }
};

std::shared_ptr<PhysicalDomain>
PhysicalDomain::clone(mesh::Block const &) const
{
    UNIMPLEMENTED;
    return std::shared_ptr<PhysicalDomain>(nullptr);
};


//void
//PhysicalDomain::run(Real stop_time, int num_of_step)
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
PhysicalDomain::print(std::ostream &os, int indent) const
{
    auto it = m_pimpl_->m_attr_.begin();
    auto ie = m_pimpl_->m_attr_.end();

    os << std::setw(indent + 1) << " " << " Type=\"" << get_class_name() << "\"," << std::endl;

    os << std::setw(indent + 1) << " " << " Mesh= {" << std::endl;

    m_mesh_->print(os, indent + 2);

    os << std::setw(indent + 1) << " " << " }," << std::endl;

//    os << std::setw(indent + 1) << " time =" << m_self_->m_time_ << ", dt =" << m_self_->m_dt_ << "," << std::endl;

    os << std::setw(indent + 1) << " " << " Attributes= {";
    for (auto const &item:m_pimpl_->m_attr_) { os << " " << item.first << ","; }
    os << "}," << std::endl;

    return os;
}


toolbox::IOStream &
PhysicalDomain::load(toolbox::IOStream &is) const
{
    if (!m_properties_["DISABLE_LOAD"]) { UNIMPLEMENTED; }
    return is;
}

toolbox::IOStream &
PhysicalDomain::save(toolbox::IOStream &os, int flag) const
{
    auto pwd = os.pwd();
//    if (!m_properties_["DISABLE_SAVE"])
    {


        for (auto const &item:m_pimpl_->m_attr_)
        {
            if (!item.second->empty())
            {
                os.open(item.first + "/");
#ifndef NDEBUG
                os.write(m_mesh_->name(), item.second->dataset(mesh::SP_ES_ALL), flag);
#else
                os.write(m_mesh_->name(), item.second->dataset(mesh::SP_ES_OWNED), flag);
#endif
                os.open(pwd);
            }
        }
    }
    if (m_properties_["DUMP_ATTR"])
    {
        auto const &attr_list = m_properties_["DUMP_ATTR"].as<std::list<std::string>>();
        for (auto const &key:attr_list)
        {
            auto it = m_pimpl_->m_attr_.find(key);
            if ((it != m_pimpl_->m_attr_.end()) && !it->second->empty())
            {
                os.open(key + "/");
#ifndef NDEBUG
                os.write(m_mesh_->name(), it->second->dataset(mesh::SP_ES_ALL), flag);
#else
                os.write(m_mesh_->name(), it->second->dataset(mesh::SP_ES_OWNED), flag);
#endif

                os.open(pwd);
            }
        }
    }
    return os;
};


void PhysicalDomain::sync(mesh::TransitionMap const &t_map, PhysicalDomain const &other)
{
    for (auto const &item:m_pimpl_->m_attr_)
    {
        if (!item.second->empty())
        {
            t_map.pull_back(item.second->data().get(), other.attribute(item.first)->data().get(),
                            item.second->entity_size_in_byte(), item.second->entity_type());
        }

    }
}

bool PhysicalDomain::same_as(mesh::Block const &) const
{
    UNIMPLEMENTED;
    return false;
};

//std::vector<mesh::box_type> PhysicalDomain::refine_boxes() const
//{
//    UNIMPLEMENTED;
//    return std::vector<mesh::box_type>();
//}
//
//void PhysicalDomain::refine(mesh::Block const &other) { UNIMPLEMENTED; };
//
//bool PhysicalDomain::coarsen(mesh::Block const &other)
//{
//    UNIMPLEMENTED;
//    return false;
//};


}} //namespace simpla { namespace simulation
