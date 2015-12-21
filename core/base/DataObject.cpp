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
    m_properties_.print(os, indent + 1);
    return os;
}

}}//namespace simpla { namespace base
