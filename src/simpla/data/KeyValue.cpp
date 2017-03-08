//
// Created by salmon on 17-3-8.
//

#include "KeyValue.h"
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "DataTraits.h"
namespace simpla {
namespace data {

KeyValue::KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p) : base_type(k, p) {}
KeyValue::KeyValue(KeyValue const& other) : base_type(other) {}
KeyValue::KeyValue(KeyValue&& other) : base_type(other) {}
KeyValue::~KeyValue() {}

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u) {
    auto res = std::make_shared<DataTable>();
    for (KeyValue const& v : u) { res->Set(v.first, v.second); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}

}  // namespace data {
}  // namespace simpla {