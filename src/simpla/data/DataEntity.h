//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/mpl/integer_sequence.h>
#include <simpla/toolbox/Any.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
#include <vector>
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
template <typename U>
struct DataEntityAdapterBase;

struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() {}
    DataEntity(DataEntity const& other) = delete;
    DataEntity(DataEntity&& other) = delete;
    virtual ~DataEntity() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; };
    virtual std::type_info const& type() const = 0;

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
//    virtual void* pointer() const { return nullptr; };

    template <typename U>
    bool isEqualTo(U const& v) const {
        if (type() != typeid(U)) { THROW_EXCEPTION_BAD_CAST(type().name(), typeid(U).name()); }
        return static_cast<DataEntityAdapterBase<U>*>(this)->equal(v);
    }

    template <typename U>
    U GetValue() const {
        if (type() != typeid(U)) { THROW_EXCEPTION_BAD_CAST(type().name(), typeid(U).name()); }
        return static_cast<DataEntityAdapterBase<U> const*>(this)->value();
    }
};
template <typename U>
struct DataEntityAdapterBase : public DataEntity {
    SP_OBJECT_HEAD(DataEntityAdapterBase<U>, DataEntity);

   public:
    DataEntityAdapterBase(){};
    virtual ~DataEntityAdapterBase(){};
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; };
    virtual std::type_info const& type() const { return typeid(U); };

    virtual bool equal(U const& v) const { return value() == v; };
    virtual U value() const = 0;
    virtual U* pointer() const = 0;
};
template <typename U, typename Enable = void>
struct DataEntityAdapter : public U, public DataEntityAdapterBase<U> {
    DataEntityAdapter(U const& v) : U(v){};
    DataEntityAdapter(U&& v) : U(v){};
    ~DataEntityAdapter() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << *static_cast<U const*>(this);
        return os;
    }
    virtual bool equal(U const& v) const { return *static_cast<U const*>(this) == v; }
    virtual U value() const { return *this; }
    virtual U* pointer() const { return const_cast<U*>(static_cast<U const*>(this)); }
};
template <typename U>
struct DataEntityAdapter<U, std::enable_if_t<std::is_arithmetic<U>::value>> : public DataEntityAdapterBase<U> {
    DataEntityAdapter(U const& v) : m_data_(v){};
    DataEntityAdapter(U&& v) : m_data_(v){};
    ~DataEntityAdapter() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << m_data_;
        return os;
    }
    virtual bool equal(U const& v) const { return m_data_ == v; }
    virtual U value() const { return m_data_; }
    virtual U* pointer() const { return &m_data_; }

   private:
    mutable U m_data_;
};

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityAdapter<std::remove_cv_t<U>>>(u));
}

inline std::shared_ptr<DataEntity> make_data_entity(char const* u) { return make_data_entity(std::string(u)); }
// template <typename V, int... N>
// struct DataEntityAdapter<nTuple<V, N...>> : public DataEntityAdapterBase<nTuple<V, N...>>, public nTuple<V, N...> {
//    DataEntityAdapter(simpla::traits::nested_initializer_list_t<V, sizeof...(N)> v) : nTuple<V, N...>(v) {}
//    ~DataEntityAdapter() {}
//    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
//        os << *static_cast<nTuple<V, N...> const*>(this);
//        return os;
//    };
//    virtual std::type_info const& type() const { return typeid(nTuple<V, N...>); }
//    virtual nTuple<V, N...> value() const { return *this; };
//    virtual void* pointer() const { return reinterpret_cast<void*>(const_cast<V*>(&(*this)[0])); };
//};
// template <typename V>
// struct DataEntityAdapter<std::vector<V>> : public DataEntityAdapterBase<std::vector<V>>, public std::vector<V> {
//    DataEntityAdapter(std::initializer_list<V> v) : std::vector<V>(v) {}
//    ~DataEntityAdapter() {}
//    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
//        //        os << *this;
//        return os;
//    };
//    virtual std::type_info const& type() const { return typeid(std::vector<V>); }
//    virtual std::vector<V> value() const { return *this; };
//    virtual void* pointer() const { return reinterpret_cast<void*>(const_cast<V*>(&(*this)[0])); };
//};
// template <typename U>
// std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u) {
//    std::shared_ptr<DataEntity> res;
//    switch (u.size()) {
//        case 1:
//            res = make_data_entity(*u.begin());
//            break;
//        case 2:
//            res = std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityAdapter<nTuple<U, 2>>>(u));
//            break;
//        case 3:
//            res = std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityAdapter<nTuple<U, 3>>>(u));
//            break;
//        default:
//            res = std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityAdapter<std::vector<U>>>(u));
//            break;
//    }
//    return res;
//}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
