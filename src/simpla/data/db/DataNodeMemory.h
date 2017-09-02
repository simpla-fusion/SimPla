//
// Created by salmon on 17-9-1.
//

#ifndef SIMPLA_DATANODEMEMORY_H
#define SIMPLA_DATANODEMEMORY_H
#include "../DataBlock.h"
#include "../DataNode.h"
namespace simpla {
namespace data {

struct DataNodeMemory : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemory, DataNode)
    SP_DATA_NODE_HEAD(DataNodeMemory)
   protected:
    explicit DataNodeMemory(eNodeType e_type) : DataNode(e_type) {}

   public:
    std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const override;

    size_type size() const override;

    size_type Set(std::string const& uri, std::shared_ptr<const DataNode> v) override;
    size_type Add(std::string const& uri, std::shared_ptr<const DataNode> v) override;
    size_type Delete(std::string const& uri) override;
    std::shared_ptr<const DataNode> Get(std::string const& uri) const override;
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const override;

    size_type Set(index_type s, std::shared_ptr<const DataNode> v) override;
    size_type Add(index_type s, std::shared_ptr<const DataNode> v) override;
    size_type Delete(index_type s) override;
    std::shared_ptr<const DataNode> Get(index_type s) const override;
    size_type Add(std::shared_ptr<const DataNode> v) override;

   private:
    std::map<std::string, std::shared_ptr<DataNode>> m_table_;
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATANODEMEMORY_H
