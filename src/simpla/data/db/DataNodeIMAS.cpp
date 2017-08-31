//
// Created by salmon on 17-8-24.
//
#include "../DataNode.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

struct DataNodeIMAS : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeIMAS, DataNode)
    SP_DATA_NODE_HEAD(DataNodeIMAS);
    SP_DATA_NODE_FUNCTION(DataNodeIMAS);

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    bool isValid() const override;

   private:
};
REGISTER_CREATOR(DataNodeIMAS, imas);
DataNodeIMAS::DataNodeIMAS(DataNode::eNodeType etype) : DataNode(etype) {}
DataNodeIMAS::~DataNodeIMAS() = default;

int DataNodeIMAS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    return 0;
}
int DataNodeIMAS::Disconnect() { return 0; }
bool DataNodeIMAS::isValid() const { return false; }

size_type DataNodeIMAS::size() const { return 0; }
std::shared_ptr<DataNode> DataNodeIMAS::CreateNode(eNodeType e_type) const { return nullptr; }

std::shared_ptr<DataEntity> DataNodeIMAS::GetEntity() const { return nullptr; }
size_type DataNodeIMAS::SetEntity(const std::shared_ptr<DataEntity>&) { return 0; }

size_type DataNodeIMAS::Set(std::string const& uri, std::shared_ptr<DataNode> const& v) { return 0; }
size_type DataNodeIMAS::Add(std::string const& uri, std::shared_ptr<DataNode> const& v) { return 0; }
size_type DataNodeIMAS::Delete(std::string const& s) { return 0; }
std::shared_ptr<DataNode> DataNodeIMAS::Get(std::string const& uri) const { return 0; }
size_type DataNodeIMAS::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const {
    return 0;
}

size_type DataNodeIMAS::Set(size_type s, std::shared_ptr<DataNode> const& v) { return 0; }
size_type DataNodeIMAS::Add(size_type s, std::shared_ptr<DataNode> const& v) { return 0; }
size_type DataNodeIMAS::Delete(size_type s) { return 0; }
std::shared_ptr<DataNode> DataNodeIMAS::Get(size_type s) const { return nullptr; }

}  // { namespace data {
}  // namespace simpla