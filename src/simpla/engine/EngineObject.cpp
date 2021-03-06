//
// Created by salmon on 17-9-3.
//

#include "EngineObject.h"
#include <simpla/data/Configurable.h>
#include <simpla/data/Serializable.h>
#include <simpla/utilities/Factory.h>
#include <boost/functional/hash.hpp>       //for uuid
#include <boost/uuid/uuid.hpp>             //for uuid
#include <boost/uuid/uuid_generators.hpp>  //for uuid
#include <mutex>
namespace simpla {
namespace engine {
struct EngineObject::pimpl_s {
    std::mutex m_mutex_;
    size_type m_click_ = 0;
    size_type m_click_tag_ = 0;
    bool m_is_initialized_ = false;
    bool m_is_setup_ = false;
};
static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

EngineObject::EngineObject() : m_pimpl_(new pimpl_s) { SetUUID(g_obj_hasher(g_uuid_generator())); }
EngineObject::~EngineObject() { Finalize(); }
std::shared_ptr<data::DataEntry> EngineObject::Serialize() const { return data::Serializable::Serialize(); }
void EngineObject::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    ASSERT(!isSetUp());
    Initialize();
    data::Serializable::Deserialize(cfg);
};

std::shared_ptr<EngineObject> EngineObject::Copy() const { return nullptr; }

void EngineObject::lock() { m_pimpl_->m_mutex_.lock(); }
void EngineObject::unlock() { m_pimpl_->m_mutex_.unlock(); }
bool EngineObject::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

size_type EngineObject::GetTagCount() const { return m_pimpl_->m_click_tag_; }
size_type EngineObject::GetClickCount() const { return m_pimpl_->m_click_; }

void EngineObject::Click() { ++m_pimpl_->m_click_; }
void EngineObject::Tag() { m_pimpl_->m_click_tag_ = m_pimpl_->m_click_; }
void EngineObject::ResetTag() { m_pimpl_->m_click_tag_ = (m_pimpl_->m_click_ = 0); }

bool EngineObject::isModified() const { return m_pimpl_->m_click_tag_ != m_pimpl_->m_click_; }
bool EngineObject::isInitialized() const { return m_pimpl_->m_is_initialized_; }
bool EngineObject::isSetUp() const { return m_pimpl_->m_is_setup_; }

// void EngineObject::Push(std::shared_ptr<data::DataEntry> const &data) { ASSERT(isSetUp()); }
// std::shared_ptr<data::DataEntry> EngineObject::Pop() const { return
// data::DataEntry::New(data::DataEntry::DN_TABLE);
// };

void EngineObject::Push(const std::shared_ptr<Patch> &) { ASSERT(isSetUp()); };
std::shared_ptr<Patch> EngineObject::Pop() const { return Patch::New(); }

void EngineObject::DoInitialize() {}
void EngineObject::DoSetUp() {}
void EngineObject::DoUpdate() {}
void EngineObject::DoTearDown() {}
void EngineObject::DoFinalize() {}

void EngineObject::Initialize() {
    if (!isInitialized()) {
        DoInitialize();
        Click();
        m_pimpl_->m_is_initialized_ = true;
    }
}
void EngineObject::SetUp() {
    if (!isSetUp()) {
        PreSetUp(this);
        DoSetUp();
        PostSetUp(this);
        Click();
        m_pimpl_->m_is_setup_ = true;
    }
}
void EngineObject::Update() {
    SetUp();
    if (isModified()) {
        PreUpdate(this);
        DoUpdate();
        PostUpdate(this);
        Tag();
    }
}
void EngineObject::TearDown() {
    if (isSetUp()) {
        PreTearDown(this);
        DoTearDown();
        PostTearDown(this);
        Click();
        m_pimpl_->m_is_setup_ = false;
    }
};
void EngineObject::Finalize() {
    if (isInitialized()) {
        TearDown();
        DoFinalize();
        ResetTag();
        m_pimpl_->m_is_initialized_ = false;
    }
};
}
}  // namespace simpla