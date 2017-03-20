//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
#include <vector>
#include "DataTraits.h"
namespace simpla {
namespace data {
template <typename, typename Enable = void>
class DataEntityWrapper {};
class DataEntity;
class DataArray;
template <typename U, typename Enable = void>
struct data_entity_traits {};
struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity();
    DataEntity(DataEntity const&) = delete;
    DataEntity(DataEntity&&) = delete;
    virtual ~DataEntity();

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << "null";
        return os;
    };
    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual bool isLight() const { return true; }
    virtual bool isHeavyBlock() const { return false; }
    virtual bool isTable() const { return false; }
    virtual bool isArray() const { return false; }
    virtual bool isNull() const { return !(isHeavyBlock() || isLight() || isTable() || isArray()); }

    virtual std::shared_ptr<DataEntity> Duplicate() const { return nullptr; };
};
template <typename U>
struct DataEntityWrapper<U, std::enable_if_t<traits::is_light_data<U>::value>> : public DataEntity {
    SP_OBJECT_HEAD(DataEntityWrapper<U>, DataEntity);
    typedef U value_type;

   public:
    DataEntityWrapper() {}
    DataEntityWrapper(value_type const& d) : m_data_(d) {}
    DataEntityWrapper(value_type&& d) : m_data_(d) {}
    virtual ~DataEntityWrapper() {}
    virtual std::type_info const& value_type_info() const { return typeid(value_type); }
    virtual bool isEntity() const { return true; }
    virtual std::shared_ptr<DataEntity> Duplicate() const { return std::make_shared<DataEntityWrapper<U>>(m_data_); };
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        if (typeid(U) == typeid(std::string)) {
            os << "\"" << value() << "\"";
        } else {
            os << value();
        }
        return os;
    }
    virtual bool equal(value_type const& other) const { return m_data_ == other; }
    virtual value_type value() const { return m_data_; }

   private:
    value_type m_data_;
};

template <typename U>
struct data_entity_traits<U, std::enable_if_t<traits::is_light_data<U>::value>> {
    static U from(DataEntity const& v) { return v.cast_as<DataEntityWrapper<U>>().value(); };
    static std::shared_ptr<DataEntity> to(U const& v) { return std::make_shared<DataEntityWrapper<U>>(v); };
};

template <typename U>
U data_cast(DataEntity const& v) {
    return data_entity_traits<U>::from(v);
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u) {
    return data_entity_traits<U>::to(u);
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u) {
    return data_entity_traits<U>::to(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) {
    return data_entity_traits<std::string>::to(std::string(u));
}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
