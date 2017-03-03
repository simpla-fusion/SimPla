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

/** @ingroup data */

/**
 * @brief primary object of data
 */
struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() {}
    template <typename T>
    DataEntity(T const& v) : m_data_(v) {}
    template <typename T>
    DataEntity(T&& v) : m_data_(v) {}
    DataEntity(DataEntity const& other) {}
    DataEntity(DataEntity&& other) {}
    virtual ~DataEntity() {}

    virtual bool isNull() const { return !(isTable() | isLight() | isHeavy()); }
    virtual bool isTable() const { return false; };
    virtual bool isLight() const { return true; };
    virtual bool isHeavy() const { return false; };

    //    virtual void* data() = 0;
    //    virtual void const* data() const = 0;

    virtual void clear() { m_data_.clear(); }
    virtual bool empty() const { return m_data_.empty(); }
    virtual std::type_info const& type() const { return m_data_.type(); }

    template <typename U>
    bool isEqualTo(U const& v) const {
        return GetValue<U>() == v;
    }
    template <typename U>
    U GetValue() const {
        return toolbox::AnyCast<U>(*this);
    }

   private:
    toolbox::Any m_data_;
};

template <typename U>
std::shared_ptr<DataEntity> make_shared_entity(U const& u) {
    return std::make_shared<DataEntity>(u);
}

template <typename U>
DataEntity make_entity(U const& u) {
    return DataEntity(u);
}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
