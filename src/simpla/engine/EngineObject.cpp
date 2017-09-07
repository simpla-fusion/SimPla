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
    bool m_is_setup_ = false;
};
EngineObject::EngineObject() : m_pimpl_(new pimpl_s) {}
EngineObject::~EngineObject() { Finalize(); }
std::shared_ptr<data::DataNode> EngineObject::Serialize() const { return base_type::Serialize(); }
void EngineObject::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    ASSERT(!isSetUp());
    Initialize();
    base_type::Deserialize(cfg);
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
bool EngineObject::isSetUp() const { return m_pimpl_->m_is_setup_; }

void EngineObject::Push(std::shared_ptr<data::DataNode> const &data) {
    ASSERT(isSetUp());
    base_type::Push(data);
    Click();
}
std::shared_ptr<data::DataNode> EngineObject::Pop() {
    Click();
    return base_type::Pop();
}

void EngineObject::DoInitialize() {}
void EngineObject::DoSetUp() {}
void EngineObject::DoUpdate() {}
void EngineObject::DoTearDown() { db()->Clear(); }
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

        VERBOSE << std::setw(15) << " Set Up : " << std::setw(20) << std::left << GetName() << " [ " << TypeName()
                << " ]";
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
        VERBOSE << std::setw(15) << " Tear Down : " << std::setw(20) << std::left << GetName() << " [ " << TypeName()
                << " ]";
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