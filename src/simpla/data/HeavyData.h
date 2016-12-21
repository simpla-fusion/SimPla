//
// Created by salmon on 16-12-6.
//

#ifndef SIMPLA_DATAENTITYHEAVY_H
#define SIMPLA_DATAENTITYHEAVY_H

#include <simpla/concept/LifeControllable.h>
#include "DataEntity.h"

namespace simpla { namespace data
{
/** @ingroup data */
/**
 * @brief  large data, which should not be passed  between modules by value, such as big matrix or
 */
struct HeavyData : public DataEntity
{
    SP_OBJECT_HEAD(HeavyData, DataEntity);
public:

    HeavyData() {}

    virtual ~HeavyData() {}

    virtual bool is_heavy() const { return true; }

    virtual void deep_copy(HeavyData const &other) {}

    virtual void clear() {}

    virtual void *data() { return nullptr; }

    virtual void const *data() const { return nullptr; }

};

}}
#endif //SIMPLA_DATAENTITYHEAVY_H
