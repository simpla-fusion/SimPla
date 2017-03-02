//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/mpl/integer_sequence.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
namespace simpla {
namespace data {

/** @ingroup data */
struct DataEntity;

template <typename U>
struct DataEntityLight;
namespace traits {
enum { LIGHT, HEAVY, TABLE };
template <typename U>
struct data_entity_traits {
    static constexpr int value = LIGHT;
};
template <typename T, typename Enable = void>
struct data_cast {};

}  // namespace traits{
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

    virtual bool isNull() const { return !(isTable() | isLight() | isHeavy()); }
    virtual bool isTable() const { return false; };
    virtual bool isLight() const { return false; };
    virtual bool isHeavy() const { return false; };

    virtual std::shared_ptr<DataEntity> Copy() const { return nullptr; };
    virtual std::shared_ptr<DataEntity> Move() { return nullptr; };
    virtual void DeepCopy(DataEntity const& other) { UNIMPLEMENTED; }

    template <typename U>
    U GetValue() const {
        return traits::data_cast<U>::value(*this);
    }
    template <typename U>
    bool isEqualTo(U const& v) const {
        return traits::data_cast<U>::value(*this) == v;
    }
};
template <typename U>
struct DataEntityLight : public DataEntity {
    typedef U value_type;
    SP_OBJECT_HEAD(DataEntityLight<U>, DataEntity);

   public:
    DataEntityLight(U const& u) : m_value_(u){};
    DataEntityLight(U&& u) : m_value_(u){};
    virtual ~DataEntityLight() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << m_value_;
        return os;
    };
    virtual bool isLight() const { return true; };
    value_type& GetValue() { return m_value_; }
    value_type const& GetValue() const { return m_value_; }

    virtual std::shared_ptr<DataEntity> Copy() const {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight<value_type>>(m_value_));
    };
    virtual std::shared_ptr<DataEntity> Move() {
        return std::dynamic_pointer_cast<DataEntity>(
            std::make_shared<DataEntityLight<value_type>>(std::move(m_value_)));
    }

   private:
    value_type m_value_;
};

namespace traits {
template <typename U>
struct data_cast<U, std::enable_if_t<traits::data_entity_traits<U>::value == traits::LIGHT>> {
    static U& value(DataEntity& v) { return v.as<DataEntityLight<U>>()->GetValue(); }
    static U const& value(DataEntity const& v) { return v.as<DataEntityLight<U>>()->GetValue(); };

    static std::shared_ptr<DataEntity> create(U const& c) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight<U>>(c));
    }
};
}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
