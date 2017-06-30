//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/SPObject.h>
#include <typeindex>
#include <vector>
#include "DataTraits.h"
#include "Serializable.h"
namespace simpla {
namespace data {
template <typename, typename Enable = void>
class DataEntityWrapper {};

class DataEntity;
class DataArray;

template <typename U, typename Enable = void>
struct data_entity_traits {};

struct DataEntity {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() = default;
    virtual ~DataEntity() = default;

    SP_DEFAULT_CONSTRUCT(DataEntity)

    virtual std::ostream& Serialize(std::ostream& os, int indent) const;
    virtual std::istream& Deserialize(std::istream& is);

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
    DataEntityWithType() = default;
    ~DataEntityWithType() = default;
    SP_DEFAULT_CONSTRUCT(DataEntityWithType)

    std::type_info const& value_type_info() const override { return typeid(value_type); }
    bool isLight() const override { return traits::is_light_data<value_type>::value; }

    //    virtual bool equal(value_type const& other) const = 0;
    virtual value_type value() const = 0;
    virtual value_type* get() { return nullptr; }
    virtual value_type const* get() const { return nullptr; }

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        if (value_type_info() == typeid(std::string)) {
            os << "\"" << value() << "\"";
        } else {
            os << value();
        }
        return os;
    };
};
template <>
struct DataEntityWrapper<void> : public DataEntity {};
template <typename U>
struct DataEntityWrapper<U> : public DataEntityWithType<U> {
    SP_OBJECT_HEAD(DataEntityWrapper<U>, DataEntityWithType<U>);
    typedef U value_type;

   public:
    DataEntityWrapper() : DataEntityWithType<U>() {}

    explicit DataEntityWrapper(std::shared_ptr<value_type> const& d) : m_data_((d)) {}
    template <typename... Args>
    DataEntityWrapper(Args&&... args) : m_data_(std::make_shared<U>(std::forward<Args>(args)...)) {}
    virtual ~DataEntityWrapper() {}

    std::type_info const& value_type_info() const override { return typeid(value_type); }

    bool isLight() const override { return traits::is_light_data<value_type>::value; }

    std::shared_ptr<DataEntity> Duplicate() const override { return std::make_shared<DataEntityWrapper<U>>(*m_data_); };

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        if (typeid(U) == typeid(std::string)) {
            os << "\"" << value() << "\"";
        } else {
            os << value();
        }
        return os;
    }
    //    bool equal(value_type const& other) const override { return *m_holder_ == other; }
    value_type value() const override { return *m_data_; };

    value_type* get() override { return m_data_.get(); }
    value_type const* get() const override { return m_data_.get(); }

   private:
    std::shared_ptr<value_type> m_data_;
};

inline std::shared_ptr<DataEntity> make_data_entity() { return nullptr; }
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u) {
    return std::make_shared<DataEntityWrapper<U>>(u);
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u,
                                             ENABLE_IF((std::is_base_of<DataEntity, U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(u);
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u,
                                             ENABLE_IF((!std::is_base_of<DataEntity, U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<U>>(u));
}
template <typename U, typename... Args>
std::shared_ptr<DataEntity> make_data_entity(Args&&... args) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<U>>(std::forward<Args>(args)...));
}

template <typename U>
struct DataCastTraits {
    static U Get(std::shared_ptr<DataEntity> const& p) {
        ASSERT(p != nullptr && p->isA<DataEntityWrapper<U>>());
        return p->cast_as<DataEntityWrapper<U>>().value();
    }
    static U Get(std::shared_ptr<DataEntity> const& p, U const& default_value) {
        ASSERT(p == nullptr || p->isA<DataEntityWrapper<U>>());
        return p == nullptr ? default_value : p->cast_as<DataEntityWrapper<U>>().value();
    }
};
template <typename U>
U data_cast(std::shared_ptr<DataEntity> const& p) {
    return DataCastTraits<U>::Get(p);
}
template <typename U>
U data_cast(std::shared_ptr<DataEntity> const& p, U const& default_value) {
    return DataCastTraits<U>::Get(p, default_value);
}

inline std::shared_ptr<DataEntity> make_data_entity(char const* u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<std::string>>(u));
}

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
