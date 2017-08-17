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
    std::shared_ptr<DataEntity> New(Args&&... args);

    template <typename U>
    static std::shared_ptr<DataEntity> New(U&& u);

    virtual std::type_info const& value_type_info() const { return typeid(void); };

    template <typename U>
    bool Check(U const& u = true) const;

    template <typename U>
    bool isA() const;
};

struct DataLight : public DataEntity {
    SP_OBJECT_HEAD(DataLight, DataEntity);
    std::experimental::any m_data_;

   protected:
    DataLight() = default;
    template <typename U>
    explicit DataLight(U const& d) : m_data_(d) {}
    template <typename U>
    explicit DataLight(U&& d) : m_data_(std::forward<U>(d)) {}

   public:
    ~DataLight() override = default;
    SP_DECLARE_NAME(DataLight);

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    std::type_info const& value_type_info() const override { return m_data_.type(); };

    size_type Count() const { return 1; }

    std::experimental::any value() const { return m_data_; };

    template <typename U>
    U as() const {
        return std::experimental::any_cast<U>(m_data_);
    }
};

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
    return DataLight::New(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) { return DataLight::New(std::string(u)); }

template <typename... Args>
std::shared_ptr<DataEntity> DataEntity::New(Args&&... args) {
    return make_data_entity(std::forward<Args>(args)...);
}

std::ostream& operator<<(std::ostream& os, DataEntity const& v);
std::istream& operator>>(std::istream& is, DataEntity& v);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
