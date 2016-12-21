//
// Created by salmon on 16-12-6.
//
#include "HeavyData.h"

namespace simpla { namespace data
{
HeavyData &DataEntity::as_heavy()
{
    ASSERT(is_heavy());
    return *static_cast<HeavyData *>(this);
}

HeavyData const &DataEntity::as_heavy() const
{
    ASSERT(is_heavy());
    return *static_cast<HeavyData const *>(this);
}


}}