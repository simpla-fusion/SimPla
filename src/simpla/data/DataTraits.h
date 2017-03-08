//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAENTITYFACTROY_H
#define SIMPLA_DATAENTITYFACTROY_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include <string>
namespace simpla {
namespace data {
class DataEntity;
class DataTable;
class DataArray;
template <typename, typename Enable = void>
class DataEntityWrapper {};
template <typename U, typename Enable = void>
class DataArrayWrapper {};
class KeyValue;

namespace traits {
template <typename U>
struct is_light_data
    : public std::integral_constant<bool, std::is_arithmetic<U>::value || std::is_same<U, bool>::value> {};

template <>
struct is_light_data<std::string> : public std::integral_constant<bool, true> {};

template <typename U, int... N>
struct is_light_data<simpla::algebra::declare::nTuple_<U, N...>> : public std::integral_constant<bool, true> {};

}  // namespace traits {

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(U const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<std::remove_cv_t<U>>>(u));
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayWrapper<U>>(u));
};
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayWrapper<void>>(u));
};
std::shared_ptr<DataEntity> make_data_entity(char const* u);
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u);
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<char const*> const& u);

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITYFACTROY_H
