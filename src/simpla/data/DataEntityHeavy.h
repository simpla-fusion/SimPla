//
// Created by salmon on 16-12-6.
//

#ifndef SIMPLA_DATAENTITYHEAVY_H
#define SIMPLA_DATAENTITYHEAVY_H

#include <simpla/concept/LifeControllable.h>
#include "DataEntity.h"

namespace simpla { namespace data
{
struct DataEntityHeavy : public DataEntity, public concept::LifeControllable
{
    SP_OBJECT_HEAD(DataEntityHeavy, DataEntity);
public:

    DataEntityHeavy() {}

    virtual ~DataEntityHeavy() {}

    virtual bool is_heavy() const { return true; }

    virtual void deep_copy(DataEntityHeavy const &other) {}

    virtual void clear() {}

    virtual void *data() { return nullptr; }

    virtual void const *data() const { return nullptr; }

};

}}
#endif //SIMPLA_DATAENTITYHEAVY_H
