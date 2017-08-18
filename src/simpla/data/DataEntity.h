//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <experimental/any>
#include <typeindex>
#include <vector>
#include "DataTraits.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
class DataLight;
class DataArray;
template <typename>
class DataArrayT;

struct DataEntity : public std::enable_shared_from_this<DataEntity> {
    SP_OBJECT_BASE(DataEntity);

   protected:
    DataEntity() = default;

   public:
    virtual ~DataEntity() = default;
    SP_DEFAULT_CONSTRUCT(DataEntity)
    template <typename... Args>
    static std::shared_ptr<DataEntity> New(Args&&... args);

    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual size_type value_type_size() const { return 0; };
    template <typename U>
    bool Check(U const& u = true) const;
};

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
    template <typename U>
    U as() const;
    virtual std::experimental::any any() const = 0;
    virtual std::shared_ptr<DataArray> asArray() const = 0;
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
    value_type value() const { return m_data_; };

    std::experimental::any any() const override { return std::experimental::any(m_data_); };
    std::shared_ptr<DataArray> asArray() const override;
};

template <typename U>
U DataLight::as() const {
    auto p = dynamic_cast<DataLightT<U> const*>(this);
    if (p == nullptr) { BAD_CAST; }
    return p->value();
}
template <typename U>
std::shared_ptr<DataLight> DataLight::New(U const& u) {
    return DataLightT<U>::New(u);
}
template <typename U>
bool DataEntity::Check(U const& u) const {
    auto p = dynamic_cast<DataLight const*>(this);
    return (p != nullptr) && p->as<U>() == u;
}
inline std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<DataEntity> const& u) { return u; }

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataLightT<U>::New(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) { return DataLightT<std::string>::New((u)); }

template <typename... Args>
std::shared_ptr<DataEntity> DataEntity::New(Args&&... args) {
    return make_data_entity(std::forward<Args>(args)...);
}

std::ostream& operator<<(std::ostream& os, DataEntity const& v);
std::istream& operator>>(std::istream& is, DataEntity& v);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
