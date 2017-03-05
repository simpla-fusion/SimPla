//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/mpl/integer_sequence.h>
#include <simpla/toolbox/Any.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>

namespace simpla {
namespace data {
class DataEntity;
/** @ingroup data */
enum { LIGHT, HEAVY, ARRAY, TABLE };
template <typename U, typename Enable = void>
struct data_entity_traits {
    static constexpr int value = HEAVY;
};
template <typename U>
struct data_entity_traits<U, std::enable_if_t<std::is_arithmetic<U>::value>> {
    static constexpr int value = LIGHT;
};
/**
 * @brief primary object of data
 */
template <typename U, typename Enable = void>
struct DataEntityProxy;

struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() {}
    DataEntity(DataEntity const& other) = delete;
    DataEntity(DataEntity&& other) = delete;
    virtual ~DataEntity() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;
    //    virtual void swap(DataEntity& other) = 0;

    //    DataEntity& operator=(DataEntity const& other) {
    //        DataEntity(other).swap(*this);
    //        return *this;
    //    }
    //    template <typename U>
    //    DataEntity& operator=(U const& other) {
    //        DataEntity(other).swap(*this);
    //        return *this;
    //    }
    //    template <typename U>
    //    DataEntity& operator=(U&& other) {
    //        DataEntity(other).swap(*this);
    //        return *this;
    //    }
    //    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return m_traits_->Print(os, indent,
    //    *this); };

    //    void swap(DataEntity& other) {
    //        std::swap(m_data_, other.m_data_);
    ////        std::swap(m_traits_, other.m_traits_);
    //    }

    bool isNull() const { return !(isTable() | isLight() | isHeavy()); }
    virtual bool isTable() const { return false; };
    virtual bool isLight() const { return false; };
    virtual bool isHeavy() const { return false; };

    //    virtual void* data() = 0;
    //    virtual void const* data() const = 0;
//    virtual void reset() = 0;
//    virtual void clear() = 0;
//    virtual bool empty() const = 0;
    virtual std::type_info const& type() const = 0;

    template <typename U>
    bool isEqualTo(U const& v) const {
        return static_cast<DataEntityProxy<U>*>(this)->equal(v);
    }

    template <typename U>
    U const& GetValue(ENABLE_IF(std::is_arithmetic<U>::value)) const {
        return static_cast<DataEntityProxy<U>*>(this)->GetValue();
    }
    template <typename U>
    U& GetValue(ENABLE_IF(std::is_arithmetic<U>::value)) {
        return static_cast<DataEntityProxy<U> const*>(this)->GetValue();
    }
};

template <typename U>
struct DataEntityProxy<U, std::enable_if_t<std::is_arithmetic<U>::value || std::is_same<U, std::string>::value>>
    : public DataEntity {
    SP_OBJECT_HEAD(DataEntityProxy<U>, DataEntity);

   public:
    DataEntityProxy(){};
    virtual ~DataEntityProxy(){};
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;
    virtual bool equal(U const& v) const = 0;
    virtual std::type_info const& type() const { return typeid(U); };

    virtual U const& GetValue() const = 0;
    virtual U& GetValue() = 0;
};
template <typename U>
struct DataEntityLight : public DataEntityProxy<U>, public toolbox::Any {
    DataEntityLight(U const& v) : toolbox::Any(v){};
    DataEntityLight(U&& v) : toolbox::Any(v){};

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << *toolbox::AnyCast<U>(this);
        return os;
    }
    virtual bool equal(U const& v) const { return *toolbox::AnyCast<U>(this) == v; }
    virtual U const& GetValue() const { return *toolbox::AnyCast<U>(this); }
    virtual U& GetValue() { return *toolbox::AnyCast<U>(this); }
};

template <typename U, int N>
struct DataEntityLightArray : public DataEntity {
    DataEntityLightArray(std::initializer_list<U> const& u) {}
    template <typename Other>
    DataEntityLightArray(std::initializer_list<std::initializer_list<Other>> const& u) {}
};
inline std::shared_ptr<DataEntity> make_data_entity(std::string const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight<std::string>>(u));
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) { return make_data_entity(std::string(u)); }
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u, ENABLE_IF((std::is_arithmetic<U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight<U>>(u));
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U&& u, ENABLE_IF((std::is_arithmetic<U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight<U>>(std::forward<U>(u)));
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u,
                                             ENABLE_IF((std::is_arithmetic<U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared(DataEntityLightArray<U, 1>(u)));
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u,
                                             ENABLE_IF((std::is_arithmetic<U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared(DataEntityLightArray<U, 2>(u)));
}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
