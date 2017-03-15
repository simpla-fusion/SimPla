/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "SPObject.h"

#include <simpla/data/DataTable.h>
#include <simpla/mpl/type_cast.h>
#include <simpla/toolbox/Log.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

namespace simpla {
struct SPObject::pimpl_s {
    pimpl_s() {}

    std::mutex m_mutex_;
    size_type m_click_ = 0;
    size_type m_click_tag_ = 0;
    boost::uuids::uuid m_id_;
    id_type m_short_id_;
    data::DataTable m_db_;
    std::string m_name_;
};

SPObject::SPObject() : m_pimpl_(new pimpl_s) {
    auto gen = boost::uuids::random_generator();
    m_pimpl_->m_id_ = boost::uuids::random_generator()();
    boost::hash<boost::uuids::uuid> hasher;
    m_pimpl_->m_short_id_ = hasher(m_pimpl_->m_id_);
    m_pimpl_->m_name_ = "unnamed";
}

SPObject::SPObject(SPObject &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}
SPObject::~SPObject() { OnDestroy(); }
data::DataTable const &SPObject::db() const { return m_pimpl_->m_db_; }
data::DataTable &SPObject::db() {
    Click();
    return m_pimpl_->m_db_;
}

std::string const &SPObject::name() const { return m_pimpl_->m_name_; }
void SPObject::name(std::string const &s) {
    Click();
    m_pimpl_->m_name_ = s;
}

id_type SPObject::id() const { return m_pimpl_->m_short_id_; }
bool SPObject::operator==(SPObject const &other) { return m_pimpl_->m_id_ == other.m_pimpl_->m_id_; }

void SPObject::lock() { m_pimpl_->m_mutex_.lock(); }
void SPObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool SPObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type SPObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type SPObject::GetClickCount() const { return m_pimpl_->m_click_; }
void SPObject::Click() { ++m_pimpl_->m_click_; }
void SPObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void SPObject::ResetTag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_ = 0; }
bool SPObject::isModified() const { return GetTagCount() != GetClickCount(); }
void SPObject::Initialize() {}
void SPObject::Finalize() {}
void SPObject::Destroy() {}

bool SPObject::Update() {
    if (!isModified()) { return false; }
    if (GetTagCount() == 0) {
        Initialize();
        OnInitialize();
    } else {
        OnChanged();
    }
    Tag();
    return true;
}
}  // namespace simpla { namespace base
