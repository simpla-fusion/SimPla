//
// Created by salmon on 17-3-8.
//
#include "DataTraits.h"
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
std::shared_ptr<DataEntity> make_data_entity(char const* u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityWrapper<std::string>>(std::string(u)));
}

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<char const*> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayWrapper<std::string>>(u));
}
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(u));
};
}
}