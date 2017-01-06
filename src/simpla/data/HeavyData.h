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

template <typename T>
struct HeavyDataAdapter : public HeavyData, public T {
    SP_OBJECT_HEAD(HeavyDataAdapter<T>, HeavyData);

   public:
    template <typename... Args>
    HeavyDataAdapter(Args&&... args) : T(std::forward<Args>(args)...) {}

    ~HeavyDataAdapter() {}

    virtual void deep_copy(HeavyData const& other) { UNIMPLEMENTED; }

    virtual void clear() { T::clear(); }

    virtual void* data() { return T::data(); }

    virtual void const* data() const { return T::data(); }

    virtual size_type ndims() const {
        UNIMPLEMENTED;
        return 0;
    }

    virtual index_type const* lower() const {
        UNIMPLEMENTED;
        return nullptr;
    };

    virtual index_type const* upper() const {
        UNIMPLEMENTED;
        return nullptr;
    };
};
template <typename T>
struct HeavyDataProxy : public HeavyData {
    SP_OBJECT_HEAD(HeavyDataProxy<T>, HeavyData);

   public:
    template <typename... Args>
    HeavyDataProxy(Args&&... args) : m_self_(std::make_shared<T>(std::forward<Args>(args)...)) {}

    HeavyDataProxy(std::shared_ptr<T> const& other) : m_self_(other) {}

    ~HeavyDataProxy() {}

    virtual void deep_copy(HeavyData const& other) { UNIMPLEMENTED; }

    virtual void clear() { m_self_->clear(); }

    virtual void* data() { return m_self_->data(); }

    virtual void const* data() const { return m_self_->data(); }

    virtual size_type ndims() const {
        UNIMPLEMENTED;
        return 0;
    }

    virtual index_type const* lower() const {
        UNIMPLEMENTED;
        return nullptr;
    };

    virtual index_type const* upper() const {
        UNIMPLEMENTED;
        return nullptr;
    };

   private:
    std::shared_ptr<T> m_self_;
};
namespace traits {
template <typename U>
struct create_entity<U, std::enable_if_t<!is_light<std::remove_cv_t<U>>::value>> {
    template <typename... Args>
    static std::shared_ptr<DataEntity> eval(Args&&... args) {
        return std::dynamic_pointer_cast<DataEntity>(
            std::make_shared<HeavyDataAdapter<U>>(std::forward<Args>(args)...));
    }

    static std::shared_ptr<DataEntity> eval(std::shared_ptr<U> const& p) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<HeavyDataProxy<U>>(p));
    }
};
}
}  // namespace data

}  // namespace simpla
#endif  // SIMPLA_DATAENTITYHEAVY_H
