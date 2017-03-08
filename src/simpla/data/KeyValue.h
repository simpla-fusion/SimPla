//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_KEYVALUE_H
#define SIMPLA_KEYVALUE_H

#include "DataArray.h"
#include "DataTable.h"
#include "DataTraits.h"

namespace simpla {
namespace data {

class KeyValue : public std::pair<std::string, std::shared_ptr<DataEntity>> {
    typedef std::pair<std::string, std::shared_ptr<DataEntity>> base_type;

   public:
    KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p);
    KeyValue(KeyValue const& other);
    KeyValue(KeyValue&& other);
    ~KeyValue();

    KeyValue& operator=(KeyValue const& other) {
        //        base_type::operator=(other);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        second = make_data_entity(u);
        return *this;
    }
    //    template <typename U>
    //    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
    //        second = make_data_entity(u);
    //        return *this;
    //    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data_entity(true)}; }
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_KEYVALUE_H
