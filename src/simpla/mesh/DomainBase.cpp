/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */
#include <simpla/toolbox/Object.h>
#include <simpla/toolbox/IOStream.h>
#include "DomainBase.h"


namespace simpla { namespace mesh
{
struct DomainBase::pimpl_s
{
    std::map<id_type, std::shared_ptr<AttributeBase> > m_attrs_;
};

DomainBase::DomainBase() : m_pimpl_(new pimpl_s) {}


DomainBase::~DomainBase() { teardown(); }
//
//void
//DomainBase::move_to(uuid id) { for (auto &item:m_pimpl_->m_attrs_) { item->second->move_to(id); }}
//
//
//std::shared_ptr<AttributeBase> DomainBase::attribute(uuid id) { return m_pimpl_->m_attrs_.at(id); };
//
//void DomainBase::add_attribute(PatchBase *attr, std::string const &s_name)
//{
//    m_pimpl_->m_attrs_.emplace(std::make_pair(s_name, attr));
//};
//
//void DomainBase::deploy() { LOGGER << "deploy problem domain [" << get_class_name() << "]" << std::endl; };
//
//void DomainBase::teardown()
//{
//    if (m != nullptr)
//    {
//        m = nullptr;
//        LOGGER << "Teardown problem domain [" << get_class_name() << "]" << std::endl;
//    }
//};
//
//std::shared_ptr<DomainBase>
//DomainBase::refine(index_box_type const &b, int n, int flag) const
//{
//    auto res = this->clone();
//    res->mesh()->refine(b, n, flag);
//    res->deploy();
//
//
//    return res;
//};
//
//
////void
////DomainBase::run(Real stop_time, int num_of_step)
////{
////    Real inc_t = (num_of_step > 0) ? ((stop_time - time()) / num_of_step) : dt();
////
////    while (stop_time - time() > inc_t)
////    {
////        next_time_step(inc_t);
////
////        sync();
////
////        time(time() + inc_t);
////    }
////    next_time_step(stop_time - time());
////    sync();
////    time(stop_time);
////}
//
//
//std::ostream &
//DomainBase::print(std::ostream &os, int indent) const
//{
//    auto it = m_pimpl_->m_attrs_.begin();
//    auto ie = m_pimpl_->m_attrs_.end();
//
//    os << std::setw(indent + 1) << " " << " Type=\"" << get_class_name() << "\"," << std::endl;
//
//    os << std::setw(indent + 1) << " " << " Mesh= {" << std::endl;
//
//    m->print(os, indent + 2);
//
//    os << std::setw(indent + 1) << " " << " }," << std::endl;
//
////    os << std::setw(indent + 1) << " time =" << m_self_->m_time_ << ", dt =" << m_self_->m_dt_ << "," << std::endl;
//
//    os << std::setw(indent + 1) << " " << " Attributes= {";
//    for (auto const &item:m_pimpl_->m_attrs_) { os << " " << item.first << ","; }
//    os << "}," << std::endl;
//
//    return os;
//}
//
//
//toolbox::IOStream &
//DomainBase::load(toolbox::IOStream &is) const
//{
//    if (!m_properties_["DISABLE_LOAD"]) { UNIMPLEMENTED; }
//    return is;
//}
//
//toolbox::IOStream &
//DomainBase::save(toolbox::IOStream &os, int flag) const
//{
//    auto pwd = os.pwd();
////    if (!m_properties_["DISABLE_SAVE"])
//    {
//
//
//        for (auto const &item:m_pimpl_->m_attrs_)
//        {
//            if (!item.second->empty())
//            {
//                os.open(item.first + "/");
//#ifndef NDEBUG
//                os.write(m->name(), item.second->dataset(), flag);
//#else
//                os.write(m->name(), item.second->dataset(SP_ES_OWNED), flag);
//#endif
//                os.open(pwd);
//            }
//        }
//    }
//    if (m_properties_["DUMP_ATTR"])
//    {
//        auto const &attr_list = m_properties_["DUMP_ATTR"].as<std::list<std::string>>();
//        for (auto const &key:attr_list)
//        {
//            auto it = m_pimpl_->m_attrs_.find(key);
//            if ((it != m_pimpl_->m_attrs_.end()) && !it->second->empty())
//            {
//                os.open(key + "/");
//#ifndef NDEBUG
//                os.write(m->name(), it->second->dataset(), flag);
//#else
//                os.write(m->name(), it->second->dataset(SP_ES_OWNED), flag);
//#endif
//
//                os.open(pwd);
//            }
//        }
//    }
//    return os;
//};
//
//
//void DomainBase::sync(TransitionMapBase const &t_map, DomainBase const &other)
//{
////    for (auto const &item:m_pimpl_->m_attrs_) { if (!item.second->empty()) { item.second->map(t_map, other); }}
//}

//bool DomainBase::same_as(MeshBlock const &) const
//{
//    UNIMPLEMENTED;
//    return false;
//};

//std::vector<box_type> DomainBase::refine_boxes() const
//{
//    UNIMPLEMENTED;
//    return std::vector<box_type>();
//}
//
//void DomainBase::refine(MeshBlock const &other) { UNIMPLEMENTED; };
//
//bool DomainBase::coarsen(MeshBlock const &other)
//{
//    UNIMPLEMENTED;
//    return false;
//};


}} //namespace simpla { namespace simulation
