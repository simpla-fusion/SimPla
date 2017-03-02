//
// Created by salmon on 17-3-2.
//
#include "KeyValue.h"
#include "DataTable.h"

namespace simpla {
namespace data {
namespace traits {
std::shared_ptr<DataEntity> data_cast<std::initializer_list<KeyValue>>::create(
    std::initializer_list<KeyValue> const& c) {
    auto p = std::make_shared<DataTable>();
    for (auto const& item : c) { p->SetValue(item); }
    return std::dynamic_pointer_cast<DataEntity>(p);
};
}  // namespace traits{
}  // namespace data{
}  // namespace simpla{
