//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKENDMEMORY_H
#define SIMPLA_DATABACKENDMEMORY_H

#include <ostream>
#include <typeindex>

#include "../DataNode.h"

namespace simpla {
namespace data {
struct DataNodeMemory : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemory, DataNode)
    SP_DATA_NODE_HEAD(DataNodeMemory);

   public:
    static std::shared_ptr<DataNode> New();

    std::shared_ptr<DataNode> CreateEntity(std::shared_ptr<DataEntity> const &v) const override;
    std::shared_ptr<DataNode> CreateTable() const override;
    std::shared_ptr<DataNode> CreateArray() const override;
    std::shared_ptr<DataNode> CreateFunction() const override;

};  // class DataBase {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
