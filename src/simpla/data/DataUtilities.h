//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATAUTILITIES_H
#define SIMPLA_DATAUTILITIES_H

#include "DataBlock.h"
#include "DataEntity.h"
#include "DataLight.h"
#include "DataNode.h"
namespace simpla {
namespace data {
//
//namespace traits {
//inline std::shared_ptr<DataEntity> data_cast(std::shared_ptr<DataEntity> const& u) { return u; }
//
//template <typename U>
//struct DataCast<U> {
//    static std::shared_ptr<DataEntity> cast(U const& u) { return DataLightT<U>::New(u); }
//};
//template <>
//struct DataCast<char const*> {
//    static std::shared_ptr<DataEntity> cast(char const* u) { return DataLightT<std::string>::New((u)); }
//};
//inline std::shared_ptr<DataEntity> data_cast(std::initializer_list<KeyValue> const& u) {
//    auto res = DataTable::New();
//    for (KeyValue const& v : u) { res->Set(v.first, v.second); }
//    return std::dynamic_pointer_cast<DataEntity>(res);
//}
//template <typename... Others>
//std::shared_ptr<DataEntity> data_cast(KeyValue const& first, Others&&... others) {
//    auto res = DataTable::New();
//    res->Set(first, std::forward<Others>(others)...);
//    return res;
//}
//
//template <typename U>
//std::shared_ptr<DataLightArray<U>> data_cast(U const* u, size_type n) {
//    return DataLightArray<U>::New(u, n);
//}
//inline std::shared_ptr<DataLightArray<std::string>> data_cast(std::initializer_list<char const*> const& u) {
//    return DataLightArray<std::string>::New(u);
//}
//template <typename U>
//std::shared_ptr<DataLightArray<U>> data_cast(std::initializer_list<U> const& u,
//                                             ENABLE_IF((traits::is_light_data<U>::value))) {
//    auto p = DataLightArray<U>::New();
//    for (auto const& item : u) { p->Add((item)); }
//    return p;
//}
//template <typename U>
//std::shared_ptr<DataArray> data_cast(std::initializer_list<U> const& u, ENABLE_IF((!traits::is_light_data<U>::value))) {
//    auto p = DataArray::New();
//    for (auto const& item : u) { p->Add(make_data_entity(item)); }
//    return p;
//}
//
//template <typename U>
//std::shared_ptr<DataArray> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
//    auto p = DataArray::New();
//    for (auto const& item : u) { p->Add(make_data_entity(item)); }
//    return p;
//}
//template <typename U>
//std::shared_ptr<DataArray> make_data_entity(
//    std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
//    auto p = DataArray::New();
//    for (auto const& item : u) { p->Add(make_data_entity(item)); }
//    return p;
//}
//}
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAUTILITIES_H
