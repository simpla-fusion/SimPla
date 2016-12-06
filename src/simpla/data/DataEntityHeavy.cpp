//
// Created by salmon on 16-12-6.
//
#include "DataEntityHeavy.h"

namespace simpla { namespace data
{
DataEntityHeavy &DataEntity::as_heavy()
{
    ASSERT(is_heavy());
    return *static_cast<DataEntityHeavy *>(this);
}

DataEntityHeavy const &DataEntity::as_heavy() const
{
    ASSERT(is_heavy());
    return *static_cast<DataEntityHeavy const *>(this);
}


}}