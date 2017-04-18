//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/sp_def.h>
#include <simpla/concept/Printable.h>
#include <simpla/utilities/Log.h>
#include <typeindex>
#include <vector>
#include "DataTraits.h"
#include "Serializable.h"
#include "simpla/algebra/nTupleExt.h"
namespace simpla {
namespace data {
template <typename, typename Enable = void>
class DataEntityWrapper {};

class DataEntity;
class DataArray;
template <typename U, typename Enable = void>
struct data_entity_traits {};
struct DataEntity : public Serializable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity();
    DataEntity(DataEntity const&) = delete;
    DataEntity(DataEntity&&) = delete;
    virtual ~DataEntity();

    virtual std::ostream& Serialize(std::ostream& os, int indent = 0) const;

    virtual bool empty() const { return true; }
    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual bool isLight() const { return false; }
    virtual bool isBlock() const { return false; }
    virtual bool isTable() const { return false; }
    virtual bool isArray() const { return false; }
    virtual bool isNull() const { return !(isBlock() || isLight() || isTable() || isArray()); }

    virtual std::shared_ptr<DataEntity> Duplicate() const { return nullptr; };
};

template <typename U>
class DataEntityWithType : public DataEntity {
    SP_OBJECT_HEAD(DataEntityWithType<U>, DataEntity);
    typedef U value_type;

   public:
    DataEntityWithType() {}
    virtual ~DataEntityWithType() {}

    virtual std::type_info const& value_type_info() const { return typeid(value_type); }

    virtual bool isLight() const { return traits::is_light_data<value_type>::value; }

    virtual bool equal(value_type const& other) const = 0;
    virtual value_type value() const = 0;
    virtual value_type* get() { return nullptr; }
    virtual value_type const* get() const { return nullptr; }
};
template <>
struct DataEntityWrapper<void> : public DataEntity {};
template <typename U>
struct DataEntityWrapper<U> : public DataEntityWithType<U> {
    SP_OBJECT_HEAD(DataEntityWrapper<U>, DataEntityWithType<U>);
    typedef U value_type;

   public:
    DataEntityWrapper() {}
    DataEntityWrapper(std::shared_ptr<value_type> const& d) : m_data_((d)) {}
    template <typename... Args>
    DataEntityWrapper(Args&&... args) : m_data_(std::make_shared<U>(std::forward<Args>(args)...)) {}
    virtual ~DataEntityWrapper() {}

    virtual std::type_info const& value_type_info() const { return typeid(value_type); }

    virtual bool isLight() const { return traits::is_light_data<value_type>::value; }

    virtual std::shared_ptr<DataEntity> Duplicate() const { return std::make_shared<DataEntityWrapper<U>>(*m_data_); };

    virtual std::ostream& Serialize(std::ostream& os, int indent = 0) const {
        if (typeid(U) == typeid(std::string)) {
            os << "\"" << value() << "\"";
        } else {
            os << value();
        }
        return os;
    }
    virtual bool equal(value_type const& other) const { return *m_data_ == other; }
    virtual value_type value() const { return *m_data_; };

    virtual value_type* get() { return m_data_.get(); }
    virtual value_type const* get() const { return m_data_.get(); }

   private:
    std::shared_ptr<value_type> m_data_;
};

inline std::shared_ptr<DataEntity> make_data_entity() { return nullptr; }
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u) {
    return std::make_shared<DataEntityWrapper<U>>(u);
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<U>>(u));
}
template <typename U, typename... Args>
std::shared_ptr<DataEntity> make_data_entity(Args&&... args) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<U>>(std::forward<Args>(args)...));
}

template <typename U>
decltype(auto) data_cast(DataEntity const& v) {
    return v.cast_as<DataEntityWrapper<U>>().value();
}

inline std::shared_ptr<DataEntity> make_data_entity(char const* u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<std::string>>(u));
}

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
