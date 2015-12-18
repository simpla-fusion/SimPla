/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */
#include <iomanip>
#include <ostream>
#include "Object.h"

namespace simpla { namespace base
{
SpObject::SpObject() : m_click_(0) { };

SpObject::SpObject(SpObject &&other) : m_click_(other.m_click_) { };

SpObject::SpObject(SpObject const &) : m_click_(0) { };

SpObject &SpObject::operator=(SpObject const &other)
{
    SpObject(other).swap(*this);
    return *this;
};

SpObject::~SpObject() { }

void SpObject::swap(SpObject &other) { std::swap(m_click_, other.m_click_); };

bool SpObject::is_a(std::type_info const &info) const { return typeid(SpObject) == info; }

std::string SpObject::get_class_name() const { return "SpObject"; }

bool SpObject::is_same(SpObject const &other) const { return this == &other; }


std::ostream &SpObject::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << this->get_class_name() << "= {";
    os << std::setw(indent) << "}," << std::endl;

    return os;
}

}}//namespace simpla { namespace base

