//
// Created by salmon on 16-12-6.
//

#ifndef SIMPLA_DATAENTITYHEAVY_H
#define SIMPLA_DATAENTITYHEAVY_H

#include <simpla/concept/LifeControllable.h>
#include <simpla/concept/Object.h>
#include "DataEntity.h"

namespace simpla {
namespace data {
/** @ingroup data */
/**
 * @brief  large data, which should not be passed  between modules by value, such as big matrix or
 */
struct HeavyData : public DataEntity {
    SP_OBJECT_HEAD(HeavyData, DataEntity);

   public:
    HeavyData() {}

    virtual ~HeavyData() {}

    virtual bool is_heavy() const { return true; }

    virtual void deep_copy(HeavyData const& other) {}

    virtual void clear() {}

    virtual std::type_info const& value_type_info() const = 0;

    virtual void* data() { return nullptr; }

    virtual void const* data() const { return nullptr; }

    virtual size_type ndims() const { return 0; }

    virtual index_type const* lower() const = 0;

    virtual index_type const* upper() const = 0;

    virtual void load(DataTable const& d){};

    virtual void save(DataTable* d) const {};
};

template <typename...>
struct HeavyDataAdapter;

template <typename T>
struct HeavyDataAdapter<T> : public HeavyData, public T {
    virtual void deep_copy(HeavyData const& other) {}

    virtual void clear() { T::clear(); }

    virtual void* data() { return T::data(); }

    virtual void const* data() const { return T::data(); }

    virtual size_type ndims() const { return 0; }

    virtual index_type const* lower() const = 0;

    virtual index_type const* upper() const = 0;
};
}
}
#endif  // SIMPLA_DATAENTITYHEAVY_H
