/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "simpla/SIMPLA_config.h"

#include "SPObject.h"

//#include <simpla/parallel/MPIComm.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_cast.h"

namespace simpla {
struct SPObject::pimpl_s {
    //    std::mutex m_mutex_;
    //    size_type m_click_ = 0;
    //    size_type m_click_tag_ = 0;

    id_type m_id_ = NULL_ID;
    bool m_is_initialized_ = false;
    std::shared_ptr<data::DataNode> m_db_ = nullptr;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

SPObject::SPObject() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = g_obj_hasher(g_uuid_generator());
    m_pimpl_->m_db_ = data::DataNode::New(data::DataNode::DN_TABLE);
    //    SetName(std::to_string(m_pimpl_->m_id_).substr(0, 15));
}
SPObject::~SPObject() { delete m_pimpl_; }
// std::shared_ptr<SPObject> SPObject::GlobalNew(std::shared_ptr<data::DataNode> const &v) {
//    std::shared_ptr<SPObject> res = nullptr;
//    auto db = data::DataNode::New();
//    if (GLOBAL_COMM.rank() == 0) {
//        res = New(v);
//        res->Serialize(db);
//        //        db->Sync();
//    } else {
//        //        db->Sync();
//        res = New(db);
//    }
//}
// std::shared_ptr<data::DataNode> SPObject::Pop() const { return m_pimpl_->m_db_; }
// void SPObject::Push(std::shared_ptr<data::DataNode> const &tdb) { m_pimpl_->m_db_->Set(tdb); }

std::shared_ptr<data::DataNode> SPObject::db() const { return m_pimpl_->m_db_; }
std::shared_ptr<data::DataNode> SPObject::db() { return m_pimpl_->m_db_; }
void SPObject::db(std::shared_ptr<data::DataNode> const &d) { m_pimpl_->m_db_ = d; };

std::shared_ptr<data::DataNode> SPObject::Serialize() const {
    auto db = data::DataNode::New(data::DataNode::DN_TABLE);
    db->Set(m_pimpl_->m_db_);
    db->SetValue("_CLASS_", TypeName());
    db->SetValue("_TYPE_", FancyTypeName());
    //    db->SetValue("Name", GetName());
    return db;
}
void SPObject::Deserialize(std::shared_ptr<data::DataNode> const &d) {
    if (d == nullptr) { return; }
    m_pimpl_->m_db_->Set(d);
    SetName(m_pimpl_->m_db_->GetValue<std::string>("Name", GetName()));
}
std::shared_ptr<SPObject> SPObject::CreateAndSync(std::shared_ptr<data::DataNode> const &) { return nullptr; }
std::shared_ptr<SPObject> SPObject::Create(std::string const &key) { return base_type::Create(key); };
std::shared_ptr<SPObject> SPObject::Create(std::shared_ptr<data::DataNode> const &tdb) {
    auto res = base_type::Create(tdb->GetValue<std::string>("_TYPE_"));
    res->Deserialize(tdb);
    return res;
};
id_type SPObject::GetGUID() const { return m_pimpl_->m_id_; }
void SPObject::SetName(std::string const &s) { m_pimpl_->m_db_->SetValue("Name", s); };
std::string SPObject::GetName() const {
    return m_pimpl_->m_db_->GetValue<std::string>("Name", std::string("unnamed_attribute"));
}

std::ostream &operator<<(std::ostream &os, SPObject const &obj) {
    std::cout << *obj.Serialize() << std::endl;
    return os;
}
std::istream &operator>>(std::istream &is, SPObject &obj) {
    obj.Deserialize(data::DataNode::New(data::DataNode::DN_TABLE, std::string(std::istreambuf_iterator<char>(is), {})));
    return is;
}

std::ostream &operator<<(std::ostream &os, std::shared_ptr<const SPObject> const &obj) {
    if (obj != nullptr) {
        os << *obj;
    } else {
        os << "<NULL OBJECT>";
    }
    return os;
}
std::istream &operator<<(std::istream &is, std::shared_ptr<SPObject> const &obj) {
    if (obj != nullptr) { is >> *obj; }
    return is;
}

}  // namespace simpla
