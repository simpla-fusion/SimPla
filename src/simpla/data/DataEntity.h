//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>

namespace simpla {
namespace data {

/** @ingroup data */
struct DataTable;
struct HeavyData;
struct LightData;

/** @ingroup data */

/**
 * @brief primary object of data
 */
struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() {}

    DataEntity(DataEntity const& other) {}

    DataEntity(DataEntity&& other) {}

    virtual ~DataEntity() {}

    virtual bool is_null() const { return !(is_table() | is_light() | is_heavy()); }

    virtual bool is_table() const { return false; };

    virtual bool is_light() const { return false; };

    virtual bool is_heavy() const { return false; };

    virtual std::shared_ptr<DataEntity> copy() const { return nullptr; };

    virtual std::shared_ptr<DataEntity> move() { return nullptr; };

    virtual void deep_copy(DataEntity const& other) { UNIMPLEMENTED; }

    DataTable& as_table();

    DataTable const& as_table() const;

    LightData& as_light();

    LightData const& as_light() const;

    HeavyData& as_heavy();

    HeavyData const& as_heavy() const;

//    template <typename U, typename... Args>
//    U& as(Args&&... args) {
//        return as_light().template as<U>(std::forward<Args>(args)...);
//    }
//
//    template <typename U, typename... Args>
//    U const& as(Args&&... args)const {
//        return as_light().template as<U>(std::forward<Args>(args)...);
//    };;
    //
    //    template <typename U, typename... Args>
    //    U const& as(Args&&... args) const;
    //
    //    template <typename U>
    //    bool equal(U const& u) const;
};

namespace traits {
template <typename>
struct is_light : public std::integral_constant<bool, true> {};
template <typename U, class Enable = void>
struct create_entity {};
}

template <typename U>
std::shared_ptr<DataEntity> create_data_entity(U& v) {
    return traits::create_entity<U>::eval(v);
};

/** @ingroup data */

// struct LightData : public DataEntity {
//    SP_OBJECT_HEAD(LightData, DataEntity);
//
//   public:
//    template <typename... Args>
//    LightData(Args&&... args) : m_data_(std::forward<Args>(args)...) {}
//
//    LightData(LightData const& other) : m_data_(other.m_data_) {}
//
//    LightData(LightData&& other) : m_data_(other.m_data_) {}
//
//    virtual ~LightData() {}
//
//    virtual bool is_light() const { return true; }
//
//    virtual std::ostream& print(std::ostream& os, int indent) const {
//        return m_data_.print(os, indent);
//    };
//
//    void swap(LightData& other) { m_data_.swap(other.m_data_); }
//
//    virtual std::shared_ptr<DataEntity> copy() const { return std::make_shared<this_type>(*this);
//    }
//
//    virtual std::shared_ptr<DataEntity> move() {
//        auto res = std::make_shared<this_type>();
//        res->swap(*this);
//        return std::dynamic_pointer_cast<DataEntity>(res);
//    }
//
//    LightData& operator=(const LightData& rhs) {
//        LightData(rhs).swap(*this);
//        return *this;
//    }
//
//    // move assignment
//    LightData& operator=(LightData&& rhs) {
//        rhs.swap(*this);
//        LightData().swap(rhs);
//        return *this;
//    }
//
//    template <typename U>
//    LightData& operator=(U const& v) {
//        m_data_ = v;
//        return *this;
//    };
//
//    template <typename U>
//    U as() {
//        return m_data_.as<U>();
//    };
//
//    template <typename U>
//    U const& as() const {
//        return m_data_.as<U>();
//    };
//
//    toolbox::any& any() { return m_data_; }
//
//    toolbox::any const& any() const { return m_data_; }
//
//    template <typename U>
//    bool equal(U const& u) const {
//        return (m_data_.is_same<U>()) && (m_data_.as<U>() == u);
//    }
//
//   private:
//    toolbox::any m_data_;
//};
//
// template <typename U, typename... Args>
// U& DataEntity::as(Args&&... args) {
//    return as_light().template as<U>(std::forward<Args>(args)...);
//}
//
// template <typename U, typename... Args>
// U const& DataEntity::as(Args&&... args) const {
//    return as_light().template as<U>(std::forward<Args>(args)...);
//}
//
// template <typename U>
// bool DataEntity::equal(U const& u) const {
//    return as_light().equal(u);
//}

// template<typename U> std::shared_ptr<DataEntity>
// create_data_entity(U const &v, ENABLE_IF((!std::is_arithmetic<U>::value)))
//{
//    UNIMPLEMENTED;
//    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<HeavyData>());
//};
/** @} */
}
}
#endif  // SIMPLA_DATAENTITY_H
