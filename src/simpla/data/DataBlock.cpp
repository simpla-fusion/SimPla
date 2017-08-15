//
// Created by salmon on 17-8-15.
//
#include "DataBlock.h"
namespace simpla {
namespace data {
DataBlock::DataBlock(std::shared_ptr<DataEntity> const& parent) : DataEntity(parent) {}

std::shared_ptr<DataBlock> DataBlock::New(std::shared_ptr<DataEntity> const& parent) {
    return DataBlockWrapper<Real>::New(parent);
};

}  // namespace data
}  // namespace simpla