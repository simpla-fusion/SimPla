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
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;
SPObject::SPObject(std::shared_ptr<data::DataEntity> const &t) : m_pimpl_(new pimpl_s), concept::Configurable(t) {
    db()->SetValue("GUID", std::to_string(g_obj_hasher(g_uuid_generator())));
}
SPObject::~SPObject() { OnDestroy(); }


id_type SPObject::GetGUID() const {
    // FIXME: work around some data backend do not support long int
    auto res = db()->GetValue<std::string>("GUID", "");
    if (res == "") {
        return NULL_ID;
    } else {
        return from_string<id_type>(res);
    }
}

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
