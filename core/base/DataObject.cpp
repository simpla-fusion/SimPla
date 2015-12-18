/**
 * @file DataObject.cpp
 * @author salmon
 * @date 2015-12-18.
 */
#include "DataObject.h"

namespace simpla { namespace base
{


std::ostream &DataObject::print(std::ostream &os, int indent) const
{
    properties().print(os, indent + 1);
    return Object::print(os, indent);
}

}}//namespace simpla { namespace base
