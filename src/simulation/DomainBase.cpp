/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "DomainBase.h"
#include "../toolbox/IOStream.h"

namespace simpla { namespace simulation
{
struct DomainBase::pimpl_s
{
//    Real m_dt_ = 0;
//    Real m_time_ = 0;
    std::map<std::string, mesh::AttributeBase *> m_attr_;
//    parallel::DistributedObject m_dist_obj_;
};

DomainBase::DomainBase() : m(nullptr), m_next_(nullptr), m_pimpl_(new pimpl_s) {}

DomainBase::DomainBase(std::shared_ptr<const mesh::MeshBase> msh) : m(msh), m_next_(nullptr),
                                                                            m_pimpl_(new pimpl_s) {};

DomainBase::~DomainBase() { teardown(); }


const mesh::AttributeBase *
DomainBase::attribute(std::string const &s_name) const
{
    return m_pimpl_->m_attr_.at(s_name);
};

void DomainBase::add_attribute(mesh::AttributeBase *attr, std::string const &s_name)
{
    m_pimpl_->m_attr_.emplace(std::make_pair(s_name, attr));
};

void DomainBase::deploy() { LOGGER << "deploy problem domain [" << get_class_name() << "]" << std::endl; };

void DomainBase::teardown()
{
    if (m != nullptr)
    {
        m = nullptr;
        LOGGER << "Teardown problem domain [" << get_class_name() << "]" << std::endl;
    }
};

std::shared_ptr<DomainBase>
DomainBase::refine(index_box_type const &b, int n, int flag) const
{
    auto res = this->clone();
    res->mesh()->refine(b, n, flag);
    res->deploy();


    return res;
};


//void
//DomainBase::run(Real stop_time, int num_of_step)
//{
//    Real inc_t = (num_of_step > 0) ? ((stop_time - time()) / num_of_step) : dt();
//
//    while (stop_time - time() > inc_t)
//    {
//        next_time_step(inc_t);
//
//        sync();
//
//        time(time() + inc_t);
//    }
//    next_time_step(stop_time - time());
//    sync();
//    time(stop_time);
//}


std::ostream &
DomainBase::print(std::ostream &os, int indent) const
{
    auto it = m_pimpl_->m_attr_.begin();
    auto ie = m_pimpl_->m_attr_.end();

    os << std::setw(indent + 1) << " " << " Type=\"" << get_class_name() << "\"," << std::endl;

    os << std::setw(indent + 1) << " " << " Mesh= {" << std::endl;

    m->print(os, indent + 2);

    os << std::setw(indent + 1) << " " << " }," << std::endl;

//    os << std::setw(indent + 1) << " time =" << m_self_->m_time_ << ", dt =" << m_self_->m_dt_ << "," << std::endl;

    os << std::setw(indent + 1) << " " << " Attributes= {";
    for (auto const &item:m_pimpl_->m_attr_) { os << " " << item.first << ","; }
    os << "}," << std::endl;

    return os;
}


toolbox::IOStream &
DomainBase::load(toolbox::IOStream &is) const
{
    if (!m_properties_["DISABLE_LOAD"]) { UNIMPLEMENTED; }
    return is;
}

toolbox::IOStream &
DomainBase::save(toolbox::IOStream &os, int flag) const
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
                os.write(m->name(), item.second->dataset(), flag);
#else
                os.write(m->name(), item.second->dataset(mesh::SP_ES_OWNED), flag);
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
                os.write(m->name(), it->second->dataset(), flag);
#else
                os.write(m->name(), it->second->dataset(mesh::SP_ES_OWNED), flag);
#endif

                os.open(pwd);
            }
        }
    }
    return os;
};


void DomainBase::sync(mesh::TransitionMapBase const &t_map, DomainBase const &other)
{
//    for (auto const &item:m_pimpl_->m_attr_) { if (!item.second->empty()) { item.second->map(t_map, other); }}
}

bool DomainBase::same_as(mesh::MeshBase const &) const
{
    UNIMPLEMENTED;
    return false;
};

//std::vector<mesh::box_type> DomainBase::refine_boxes() const
//{
//    UNIMPLEMENTED;
//    return std::vector<mesh::box_type>();
//}
//
//void DomainBase::refine(mesh::MeshBase const &other) { UNIMPLEMENTED; };
//
//bool DomainBase::coarsen(mesh::MeshBase const &other)
//{
//    UNIMPLEMENTED;
//    return false;
//};


}} //namespace simpla { namespace simulation
