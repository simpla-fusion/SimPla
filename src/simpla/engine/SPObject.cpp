/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "SPObject.h"

//->append/data/DataTable.h>
#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_cast.h"
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

namespace simpla {
namespace engine {
struct SPObject::pimpl_s {
    std::mutex m_mutex_;
    size_type m_click_ = 1;
    size_type m_click_tag_ = 0;
    id_type m_id_ = NULL_ID;
    std::string m_name_;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;
SPObject::SPObject(std::string const &s_name) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_id_ = g_obj_hasher(g_uuid_generator());
    m_pimpl_->m_name_ = (s_name != "") ? s_name : std::to_string(m_pimpl_->m_id_);
}
SPObject::~SPObject() { Finalize(); }
SPObject::SPObject(SPObject const &other) {}
SPObject::SPObject(SPObject &&other) {}
void SPObject::swap(SPObject &other) {}

void SPObject::SetGUID(id_type id) { m_pimpl_->m_id_ = id; }
id_type SPObject::GetGUID() const { return m_pimpl_->m_id_; }

void SPObject::SetName(std::string const &s_name) {
    m_pimpl_->m_name_ = s_name;
    Click();
}
std::string const &SPObject::GetName() const { return m_pimpl_->m_name_; }

// DataPack SPObject::Serialize() const { return DataPack{}; }
// void SPObject::UnPack(engine::DataPack &&t) {}

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

void SPObject::DoInitialize() {}
void SPObject::DoFinalize() {}
void SPObject::DoTearDown() {}
void SPObject::DoUpdate() {}

void SPObject::Initialize() {
    if (!isInitialized()) {
        PreInitialize(this);
        DoInitialize();
        PostInitialize(this);
        Click();
        Tag();
    }
}

void SPObject::Update() {
    if (isModified()) {
        Initialize();
        PreUpdate(this);
        DoUpdate();
        PostUpdate(this);
        Tag();
    }
}
void SPObject::TearDown() {
    PreTearDown(this);
    DoTearDown();
    PostTearDown(this);
    Tag();
    Click();
};
void SPObject::Finalize() {
    if (isInitialized()) {
        TearDown();
        PreFinalize(this);
        DoFinalize();
        PostFinalize(this);
        ResetTag();
    }
};
}  // namespace engine{
}  // namespace simpla { namespace base
