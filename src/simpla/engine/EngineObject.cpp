//
// Created by salmon on 17-9-3.
//

#include "EngineObject.h"
namespace simpla {
namespace engine {
struct EngineObject::pimpl_s {
    std::mutex m_mutex_;
    size_type m_click_ = 0;
    size_type m_click_tag_ = 0;
    bool m_is_initialized_ = false;
};

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

void EngineObject::DoInitialize() {}
void EngineObject::DoFinalize() {}
void EngineObject::DoTearDown() {}
void EngineObject::DoUpdate() {}

void EngineObject::Initialize() {
    if (!isInitialized()) {
        //        VERBOSE << "Initialize \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
        PreInitialize(this);
        DoInitialize();
        PostInitialize(this);
        Click();
        m_pimpl_->m_is_initialized_ = true;
    }
}
void EngineObject::Update() {
    Initialize();
    if (isModified()) {
        //        VERBOSE << "Update    \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
        PreUpdate(this);
        DoUpdate();
        PostUpdate(this);
        Tag();
    }
}
void EngineObject::TearDown() {
    if (isInitialized()) {
        PreTearDown(this);
        DoTearDown();
        PostTearDown(this);
        Click();
        //        VERBOSE << "TearDown  \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
    }
};
void EngineObject::Finalize() {
    if (isInitialized()) {
        TearDown();
        PreFinalize(this);
        DoFinalize();
        PostFinalize(this);
        ResetTag();
        //        m_pimpl_->m_is_initialized_ = false;
        //        VERBOSE << "Finalize  \t:" << GetName() << "[" << GetTypeName() << "]" << std::endl;
    }
};
}
}  // namespace simpla