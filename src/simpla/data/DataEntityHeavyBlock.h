//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATAENTITYHEAVY_H
#define SIMPLA_DATAENTITYHEAVY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/Array.h>
#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/utilities/Log.h>
#include <typeindex>
#include <vector>
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {

template <typename U, int N>
struct DataEntityWrapper<Array<U, N>> : public DataEntity, public Array<U, N> {
    typedef Array<U, N> array_type;
    SP_OBJECT_HEAD(DataEntityWrapper<array_type>, DataEntity);

   public:
    using array_type::value_type;
    using array_type::ndims;

    DataEntityWrapper() {}
    DataEntityWrapper(array_type const& d) : m_data_(d) {}
    DataEntityWrapper(array_type&& d) : m_data_(d) {}
    virtual ~DataEntityWrapper() {}
    virtual std::type_info const& value_type_info() const { return typeid(array_type); }
    virtual bool isEntity() const { return true; }
    virtual bool isLight() const { return false; }
    virtual bool isBlock() const { return false; }

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
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<U>>(u));
}
template <typename U, int N>
std::shared_ptr<DataEntity> make_data_entity(Array<U, N> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<Array<U, N>>>(u));
}
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITYHEAVY_H
