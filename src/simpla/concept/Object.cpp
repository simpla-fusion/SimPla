/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "Object.h"

#include <iomanip>
#include <ostream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/functional/hash.hpp>


#include <simpla/mpl/type_cast.h>


namespace simpla
{
struct Object::pimpl_s
{
    std::mutex m_mutex_;
    size_t m_click_ = 0;
    boost::uuids::uuid m_id_;

    id_type m_short_id_;
};

Object::Object() : m_pimpl_(new pimpl_s)
{
    auto gen = boost::uuids::random_generator();
    m_pimpl_->m_id_ = boost::uuids::random_generator()();
    boost::hash<boost::uuids::uuid> hasher;
    m_pimpl_->m_short_id_ = hasher(m_pimpl_->m_id_);

    this->touch();
}

Object::Object(Object &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}

Object::~Object() {}

id_type Object::id() const { return m_pimpl_->m_short_id_; }

void Object::id(id_type t_id) { m_pimpl_->m_short_id_ = t_id; }

bool Object::operator==(Object const &other) { return m_pimpl_->m_id_ == other.m_pimpl_->m_id_; }

void Object::lock() { m_pimpl_->m_mutex_.lock(); }

void Object::unlock() { m_pimpl_->m_mutex_.unlock(); }

bool Object::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

void Object::touch() { GLOBAL_CLICK_TOUCH(&m_pimpl_->m_click_); }

size_type Object::click() const { return m_pimpl_->m_click_; }

}//namespace simpla { namespace base

