//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATAUTILITIES_H
#define SIMPLA_DATAUTILITIES_H

#include "DataBlock.h"
#include "DataEntity.h"
#include "DataLight.h"

namespace simpla {
namespace data {

std::ostream& operator<<(std::ostream& os, DataEntity const& v);
std::istream& operator>>(std::istream& is, DataEntity& v);

inline std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<DataEntity> const& u) { return u; }

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataLightT<U>::New(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(char const* u) { return DataLightT<std::string>::New((u)); }

inline std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u) {
    auto res = DataTable::New();
    for (KeyValue const& v : u) { res->Set(v.first, v.second); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
template <typename... Others>
std::shared_ptr<DataEntity> make_data_entity(KeyValue const& first, Others&&... others) {
    auto res = DataTable::New();
    res->Set(first, std::forward<Others>(others)...);
    return res;
}

template <typename U>
std::shared_ptr<DataArray> make_data_entity(U const* u, size_type n) {
    return DataArrayT<U>::New(u, n);
}
inline std::shared_ptr<DataArrayT<std::string>> make_data_entity(std::initializer_list<char const*> const& u) {
    return DataArrayT<std::string>::New(u);
}
template <typename U>
std::shared_ptr<DataArrayT<U>> make_data_entity(std::initializer_list<U> const& u,
                                                ENABLE_IF((traits::is_light_data<U>::value))) {
    auto p = DataArrayT<U>::New();
    for (auto const& item : u) { p->Add((item)); }
    return p;
}
template <typename U>
std::shared_ptr<DataArrayT<void>> make_data_entity(std::initializer_list<U> const& u,
                                                   ENABLE_IF((!traits::is_light_data<U>::value))) {
    auto p = DataArrayT<void>::New();
    for (auto const& item : u) { p->Add(make_data_entity(item)); }
    return p;
}

template <typename U>
std::shared_ptr<DataArray> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    return DataArrayT<void>::New(u);
}
template <typename U>
std::shared_ptr<DataArray> make_data_entity(
    std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    return DataArrayT<void>::New(u);
}

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAUTILITIES_H