//
// Created by salmon on 16-12-6.
//
#include "HeavyData.h"

namespace simpla { namespace data
{
HeavyData &DataEntity::asHeavy()
{
    ASSERT(isHeavy());
    return *static_cast<HeavyData *>(this);
}

HeavyData const &DataEntity::asHeavy() const
{
    ASSERT(isHeavy());
    return *static_cast<HeavyData const *>(this);
}


}}