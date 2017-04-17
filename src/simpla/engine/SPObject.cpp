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
    id_type m_id_ = NULL_ID;
    std::string m_name_ = "unnamed";
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;
SPObject::SPObject() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = g_obj_hasher(g_uuid_generator());
    m_pimpl_->m_name_ = "";  // std::to_string(m_pimpl_->m_id_);
}
SPObject::~SPObject() { OnFinalize(); }

id_type SPObject::GetGUID() const { return m_pimpl_->m_id_; }

void SPObject::SetName(std::string const& s) { m_pimpl_->m_name_ = s; }
std::string const& SPObject::GetName() const { return m_pimpl_->m_name_; }

void SPObject::lock() { m_pimpl_->m_mutex_.lock(); }
void SPObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool SPObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type SPObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type SPObject::GetClickCount() const { return m_pimpl_->m_click_; }
void SPObject::Click() { ++m_pimpl_->m_click_; }
void SPObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void SPObject::ResetTag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_ = 0; }
bool SPObject::isModified() const { return GetTagCount() != GetClickCount(); }
void SPObject::Initialize() {
    Click();
    Tag();
}
void SPObject::Finalize() { ResetTag(); }
void SPObject::TearDown() {
    Click();
    Tag();
}
void SPObject::SetUp() {
    if (GetTagCount() == 0) {
        Initialize();
        OnInitialize();
    }
    OnSetUp();

    Tag();
}
}  // namespace simpla { namespace base
