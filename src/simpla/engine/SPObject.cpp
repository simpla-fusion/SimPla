/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "SPObject.h"

#include <simpla/data/DataTable.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/type_cast.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

namespace simpla {
struct SPObject::pimpl_s {
    std::mutex m_mutex_;
    size_type m_click_ = 1;
    size_type m_click_tag_ = 0;
    id_type m_id_ = NULL_ID;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;
SPObject::SPObject() : m_pimpl_(new pimpl_s) { m_pimpl_->m_id_ = g_obj_hasher(g_uuid_generator()); }
SPObject::~SPObject() { DoFinalize(); }

void SPObject::SetGUID(id_type id) { m_pimpl_->m_id_ = id; }

id_type SPObject::GetGUID() const { return m_pimpl_->m_id_; }

void SPObject::lock() { m_pimpl_->m_mutex_.lock(); }
void SPObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool SPObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type SPObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type SPObject::GetClickCount() const { return m_pimpl_->m_click_; }
void SPObject::Click() { ++m_pimpl_->m_click_; }
void SPObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void SPObject::ResetTag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_ = 0; }
bool SPObject::isModified() const { return m_pimpl_->m_click_tag_ != m_pimpl_->m_click_; }
bool SPObject::isInitialized() const { return m_pimpl_->m_click_tag_ > 0; }

void SPObject::Initialize() {}
void SPObject::Finalize() {}
void SPObject::TearDown() {}
void SPObject::SetUp() {}

void SPObject::DoInitialize() {
    if (!isInitialized()) {
        PreInitialize(this);
        Initialize();
        PostInitialize(this);
        Click();
        Tag();
    }
}

void SPObject::DoSetUp() {
    if (isModified()) {
        DoInitialize();
        PreSetUp(this);
        SetUp();
        PostSetUp(this);
        Tag();
    }
}
void SPObject::DoTearDown() {
    PreTearDown(this);
    TearDown();
    PostTearDown(this);
    Tag();
    Click();
};
void SPObject::DoFinalize() {
    if (isInitialized()) {
        DoTearDown();
        PreFinalize(this);
        Finalize();
        PostFinalize(this);
        ResetTag();
    }
};
}  // namespace simpla { namespace base
