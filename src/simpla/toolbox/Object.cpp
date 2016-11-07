/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "Object.h"
#include "type_cast.h"

#include <iomanip>
#include <ostream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/functional/hash.hpp>

namespace simpla { namespace toolbox
{
struct Object::pimpl_s
{
    std::string m_name_{""};
    std::mutex m_mutex_;
    size_t m_click_ = 0;
    boost::uuids::uuid m_id_;

    id_type m_short_id_;
};

Object::Object(std::string const &s) : m_pimpl_(new pimpl_s)
{
    auto gen = boost::uuids::random_generator();
    m_pimpl_->m_id_ = boost::uuids::random_generator()();
    boost::hash<boost::uuids::uuid> hasher;
    m_pimpl_->m_short_id_ = hasher(m_pimpl_->m_id_);
    m_pimpl_->m_name_ = s != "" ? s : get_class_name() + "_" + string_cast(id());;

    this->touch();
}

Object::Object(Object &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}

Object::~Object() {}


bool Object::is_a(std::type_info const &info) const { return typeid(Object) == info; }

std::string Object::get_class_name() const { return "Object"; }

std::string const &Object::name() const { return m_pimpl_->m_name_; };


id_type Object::id() const { return m_pimpl_->m_short_id_; }

bool Object::operator==(Object const &other) { return m_pimpl_->m_id_ == other.m_pimpl_->m_id_; }

void Object::lock() { m_pimpl_->m_mutex_.lock(); }

void Object::unlock() { m_pimpl_->m_mutex_.unlock(); }

bool Object::try_lock() { return m_pimpl_->m_mutex_.try_lock(); }

void Object::touch() { GLOBAL_CLICK_TOUCH(&m_pimpl_->m_click_); }

size_type Object::click() const { return m_pimpl_->m_click_; }

}}//namespace simpla { namespace base

