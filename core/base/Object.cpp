/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */
#include <iomanip>
#include <ostream>
#include "Object.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

namespace simpla { namespace base
{
Object::Object() : m_uuid_(boost::uuids::random_generator()())
{
    this->touch();

};

Object::Object(Object &&other) : m_click_(other.m_click_), m_uuid_(other.m_uuid_) { };

Object::Object(Object const &) : m_uuid_(boost::uuids::random_generator()()) { this->touch(); };

Object &Object::operator=(Object const &other)
{
    Object(other).swap(*this);
    return *this;
};

Object::~Object() { }

void Object::swap(Object &other)
{
    std::swap(m_click_, other.m_click_);
    std::swap(m_uuid_, other.m_uuid_);
};

bool Object::is_a(std::type_info const &info) const { return typeid(Object) == info; }

std::string Object::get_class_name() const { return "base::LuaObject"; }

std::ostream &Object::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << this->get_class_name() << "= {";
    os << std::setw(indent) << "}," << std::endl;

    return os;
}

}}//namespace simpla { namespace base

