/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "simpla/SIMPLA_config.h"

#include "SPObject.h"

#include <simpla/data/DataTable.h>
#include <simpla/parallel/MPIComm.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

#include "simpla/data/db/DataBaseStdIO.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_cast.h"

namespace simpla {
namespace engine {
struct SPObject::pimpl_s {
    std::mutex m_mutex_;
    size_type m_click_ = 0;
    size_type m_click_tag_ = 0;
    id_type m_id_ = NULL_ID;

    bool m_is_initialized_ = false;
    std::string m_name_;

    std::shared_ptr<data::DataTable> m_db_ = nullptr;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

SPObject::SPObject() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = g_obj_hasher(g_uuid_generator());
    m_pimpl_->m_db_ = data::DataTable::New();
}
SPObject::~SPObject() {
    Finalize();
    delete m_pimpl_;
}
static std::shared_ptr<SPObject> SPObject::GlobalNew(std::shared_ptr<const data::DataNode> const &v) {
    std::shared_ptr<SPObject> res = nullptr;
    auto db = engine::DataTable::New();
    if (GLOBAL_COMM.rank() == 0) {
        res = New(v);
        res->Serialize(db);
        db->Sync();
    } else {
        db->Sync();
        res = New(db);
    }
}

const data::DataTable &SPObject::db() const { return *m_pimpl_->m_db_; }
data::DataTable &SPObject::db() { return *m_pimpl_->m_db_; }

void SPObject::Serialize(const std::shared_ptr<data::DataNode> &cfg) const {}
void SPObject::Deserialize(const std::shared_ptr<const data::DataNode> &cfg) {}

id_type SPObject::GetGUID() const { return m_pimpl_->m_id_; }
void SPObject::SetName(std::string const &s) { m_pimpl_->m_name_ = s; };
std::string const &SPObject::GetName() const { return m_pimpl_->m_name_; }

void SPObject::lock() { m_pimpl_->m_mutex_.lock(); }
void SPObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool SPObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type SPObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type SPObject::GetClickCount() const { return m_pimpl_->m_click_; }

void SPObject::Click() { ++m_pimpl_->m_click_; }
void SPObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void SPObject::ResetTag() { m_pimpl_->m_click_tag_ = (m_pimpl_->m_click_ = 0); }
bool SPObject::isModified() const { return m_pimpl_->m_click_tag_ != m_pimpl_->m_click_; }
bool SPObject::isInitialized() const { return m_pimpl_->m_is_initialized_; }

void SPObject::DoInitialize() {}
void SPObject::DoFinalize() {}
void SPObject::DoTearDown() {}
void SPObject::DoUpdate() {}

void SPObject::Initialize() {
    if (!isInitialized()) {
        //        VERBOSE << "Initialize \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
        PreInitialize(this);
        DoInitialize();
        PostInitialize(this);
        Click();
        m_pimpl_->m_is_initialized_ = true;
    }
}
void SPObject::Update() {
    Initialize();
    if (isModified()) {
        //        VERBOSE << "Update    \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
        PreUpdate(this);
        DoUpdate();
        PostUpdate(this);
        Tag();
    }
}
void SPObject::TearDown() {
    if (isInitialized()) {
        PreTearDown(this);
        DoTearDown();
        PostTearDown(this);
        Click();
        //        VERBOSE << "TearDown  \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
    }
};
void SPObject::Finalize() {
    if (isInitialized()) {
        TearDown();
        PreFinalize(this);
        DoFinalize();
        PostFinalize(this);
        ResetTag();
        m_pimpl_->m_is_initialized_ = false;
        //        VERBOSE << "Finalize  \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
    }
};

std::ostream &operator<<(std::ostream &os, SPObject const &obj) {
    auto db = data::DataNode::New();
    obj.Serialize(db);
    std::cout << *db << std::endl;
    return os;
}
std::istream &operator>>(std::istream &is, SPObject &obj) {
    obj.Deserialize(data::DataNode::New(std::string(std::istreambuf_iterator<char>(is), {})));
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

}  // namespace engine{
}  // namespace simpla { namespace base
