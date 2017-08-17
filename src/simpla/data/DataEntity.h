//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <experimental/any>
#include <typeindex>
#include <vector>
#include "DataEntityVisitor.h"
#include "DataTraits.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
class DataLight;
enum DataEntityType { DB_NULL = 0, DB_LIGHT = 1, DB_BLOCK = 2, DB_ARRAY = 3, DB_TABLE = 4 };
struct DataEntity : public std::enable_shared_from_this<DataEntity> {
    SP_OBJECT_BASE(DataEntity);

   protected:
    DataEntity() = default;

   public:
    virtual ~DataEntity() = default;
    SP_DEFAULT_CONSTRUCT(DataEntity)
    template <typename... Args>
    static std::shared_ptr<DataEntity> New(Args&&... args);

    template <typename U>
    static std::shared_ptr<DataEntity> New(U&& u);

    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual size_type value_type_size() const { return 0; };

    template <typename U>
    bool Check(U const& u = true) const;

    template <typename U>
    bool isA() const;
};
template <typename V>
class DataLightT;

struct DataLight : public DataEntity {
    SP_OBJECT_HEAD(DataLight, DataEntity);

   protected:
    DataLight() = default;

   public:
    ~DataLight() override = default;
    SP_DEFAULT_CONSTRUCT(DataLight);

    template <typename U>
    static std::shared_ptr<this_type> New(U const& u);

    std::type_info const& value_type_info() const override = 0;
    size_type value_type_size() const override = 0;

    virtual std::experimental::any any() const = 0;

    template <typename U>
    U as() const {
        auto p = dynamic_cast<DataLightT<U> const*>(this);
        if (p == nullptr) { BAD_CAST; }
        return p->value();
    }
};
template <typename V>
class DataLightT : public DataLight {
    SP_OBJECT_HEAD(DataLightT, DataLight);
    typedef V value_type;
    value_type m_data_;

   protected:
    DataLightT() = default;
    template <typename... Args>
    explicit DataLightT(Args&&... args) : m_data_(std::forward<Args>(args)...) {}

   public:
    ~DataLightT() override = default;

    SP_DEFAULT_CONSTRUCT(DataLightT);

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    std::type_info const& value_type_info() const override { return typeid(V); };
    size_type value_type_size() const override { return sizeof(value_type); };

    std::experimental::any any() const override { return std::experimental::any(m_data_); };

    value_type value() const { return m_data_; };
};

template <typename U>
std::shared_ptr<DataLight> DataLight::New(U const& u) {
    return DataLightT<U>::New(u);
}
template <typename U>
bool DataEntity::Check(U const& u) const {
    auto p = dynamic_cast<DataLight const*>(this);
    return (p != nullptr) && p->as<U>() == u;
}

template <typename U>
bool DataEntity::isA() const {
    return value_type_info() == typeid(U);
};

template <typename U>
std::shared_ptr<DataEntity> DataEntity::New(U&& u) {
    return DataLight::New(std::forward<U>(u));
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataLightT<U>::New(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) {
    return DataLightT<std::string>::New(std::string(u));
}

template <typename... Args>
std::shared_ptr<DataEntity> DataEntity::New(Args&&... args) {
    return make_data_entity(std::forward<Args>(args)...);
}

std::ostream& operator<<(std::ostream& os, DataEntity const& v);
std::istream& operator>>(std::istream& is, DataEntity& v);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
