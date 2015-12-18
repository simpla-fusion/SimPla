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

std::ostream &SpObject::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << this->get_class_name() << "= {";
    os << std::setw(indent) << "}," << std::endl;

    return os;
}


}//namespace simpla { namespace base

