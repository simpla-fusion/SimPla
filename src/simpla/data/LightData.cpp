//
// Created by salmon on 17-1-6.
//
#include "LightData.h"
#include "DataEntity.h"

namespace simpla {
namespace data {

std::shared_ptr<DataEntity> create_data_entity(char const* v) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<LightData>(std::string(v)));
};
}  // namespace data{
}  // namespace simpla{