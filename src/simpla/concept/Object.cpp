/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "Object.h"

#include <simpla/mpl/type_cast.h>
#include <simpla/toolbox/LifeClick.h>
#include <simpla/toolbox/Log.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

namespace simpla {
struct Object::pimpl_s {
    std::mutex m_mutex_;
    size_t m_click_ = 0;
    boost::uuids::uuid m_id_;

    id_type m_short_id_;
};

Object::Object() : m_pimpl_(new pimpl_s), m_state_(NULL_STATE) {
    auto gen = boost::uuids::random_generator();
    m_pimpl_->m_id_ = boost::uuids::random_generator()();
    boost::hash<boost::uuids::uuid> hasher;
    m_pimpl_->m_short_id_ = hasher(m_pimpl_->m_id_);

    this->touch();
}

Object::Object(Object &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}

Object::~Object() { Destroy(); }

id_type Object::id() const { return m_pimpl_->m_short_id_; }

void Object::id(id_type t_id) { m_pimpl_->m_short_id_ = t_id; }

bool Object::operator==(Object const &other) { return m_pimpl_->m_id_ == other.m_pimpl_->m_id_; }

void Object::lock() { m_pimpl_->m_mutex_.lock(); }

void Object::unlock() { m_pimpl_->m_mutex_.unlock(); }

bool Object::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

void Object::touch() { GLOBAL_CLICK_TOUCH(&m_pimpl_->m_click_); }

size_type Object::click() const { return m_pimpl_->m_click_; }

void Object::Deploy() {
    if (m_state_ == NULL_STATE) { m_state_ = BLANK; }
};

void Object::Initialize() {
    if (m_state_ < BLANK) { Deploy(); }
    if (m_state_ == BLANK) {
        m_state_ = VALID;
    } else {
        RUNTIME_ERROR << "Initialize should be invoked ONLY ONCE!" << std::endl;
    }
};

void Object::PreProcess() {
    if (m_state_ < VALID) { Initialize(); }
    if (m_state_ == VALID) { m_state_ = READY; }
}

void Object::Lock() {
    // FIXME: this place should be atomic
    if (m_state_ < READY) { PreProcess(); }
    if (m_state_ == READY) { m_state_ = LOCKED; }
}
bool Object::TryLock() {
    UNIMPLEMENTED;
    return true;
}
void Object::Unlock() {
    // FIXME: this place should be atomic
    if (m_state_ < READY) { PreProcess(); }
    if (m_state_ == READY) { m_state_ = LOCKED; }
}

void Object::PostProcess() {
    if (m_state_ == READY) { m_state_ = VALID; }
}

void Object::Finalize() {
    if (m_state_ > VALID) { PostProcess(); }
    if (m_state_ == VALID) {
        m_state_ = BLANK;
    } else {
        RUNTIME_ERROR << "Finalize should be  invoked ONLY ONCE!" << std::endl;
    }
}

void Object::Destroy() {
    if (m_state_ > BLANK) { Finalize(); }
    if (m_state_ == BLANK) { m_state_ = NULL_STATE; }
};

}  // namespace simpla { namespace base
