//
// Created by salmon on 17-3-8.
//
#include "KeyValue.h"
#include "DataArray.h"
#include "DataTable.h"
namespace simpla {
namespace data {
KeyValue::KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p) : base_type(k, p) {}
KeyValue::KeyValue(KeyValue const& other) : base_type(other) {}
KeyValue::KeyValue(KeyValue&& other) : base_type(other) {}
KeyValue::~KeyValue() {}

}  // namespace data {
}  // namespace simpla {