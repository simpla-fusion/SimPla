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
    std::shared_ptr<data::DataTable> m_db_;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;
SPObject::SPObject(std::shared_ptr<data::DataEntity> const &t) : m_pimpl_(new pimpl_s) {
    auto gen = boost::uuids::random_generator();

    m_pimpl_->m_db_ = (t != nullptr && t->isTable()) ? std::dynamic_pointer_cast<data::DataTable>(t)
                                                     : std::make_shared<data::DataTable>();
    if (t != nullptr && t->isLight() && t->value_type_info() == typeid(std::string)) {
        m_pimpl_->m_db_->SetValue("name", data::data_cast<std::string>(*t));
    }
    m_pimpl_->m_db_->SetValue("GUID", g_obj_hasher(g_uuid_generator()));
}
SPObject::~SPObject() { OnDestroy(); }
std::shared_ptr<data::DataTable> SPObject::db(std::string const &uri) const {
    return uri == "" ? m_pimpl_->m_db_ : m_pimpl_->m_db_->GetTable(uri);
}
std::shared_ptr<data::DataTable> SPObject::db(std::string const &uri) {
    Click();
    return uri == "" ? m_pimpl_->m_db_ : m_pimpl_->m_db_->GetTable(uri + "/");
}

std::string SPObject::name() const { return db()->GetValue<std::string>("name", ""); }
id_type SPObject::GetGUID() const { return db()->GetValue<id_type>("GUID", NULL_ID); }


void SPObject::lock() { m_pimpl_->m_mutex_.lock(); }
void SPObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool SPObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type SPObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type SPObject::GetClickCount() const { return m_pimpl_->m_click_; }
void SPObject::Click() { ++m_pimpl_->m_click_; }
void SPObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void SPObject::ResetTag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_ = 0; }
bool SPObject::isModified() const { return GetTagCount() != GetClickCount(); }
void SPObject::Initialize() { Tag(); }
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
