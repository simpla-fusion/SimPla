//
// Created by salmon on 17-8-24.
//

#include "DataNodeIMAS.h"
namespace simpla {
namespace data {
std::shared_ptr<DataNode> DataNodeIMAS::Duplicate() const { return nullptr; }
size_type DataNodeIMAS::GetNumberOfChildren() const { return 0; }
int DataNodeIMAS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    return 0;
}
int DataNodeIMAS::Disconnect() { return 0; }
bool DataNodeIMAS::isValid() const { return false; }
int DataNodeIMAS::Flush() { return 0; }

DataNode::e_NodeType DataNodeIMAS::NodeType() const { return DN_NULL; }

std::shared_ptr<DataNode> DataNodeIMAS::Root() const { return nullptr; }
std::shared_ptr<DataNode> DataNodeIMAS::Parent() const { return nullptr; }

int DataNodeIMAS::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) { return 0; }
int DataNodeIMAS::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const { return 0; }

std::shared_ptr<DataNode> DataNodeIMAS::GetNode(std::string const& uri, int flag) { return nullptr; }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(std::string const& uri, int flag) const { return nullptr; }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(index_type s, int flag) { return nullptr; }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(index_type s, int flag) const { return nullptr; }
int DataNodeIMAS::DeleteNode(std::string const& uri, int flag) { return 0; }
void DataNodeIMAS::Clear() {}
std::shared_ptr<DataEntity> DataNodeIMAS::Get() const { return nullptr; }
int DataNodeIMAS::Set(std::shared_ptr<DataEntity> const& v) { return 0; }
int DataNodeIMAS::Add(std::shared_ptr<DataEntity> const& v) { return 0; }

}  // { namespace data {
}  // namespace simpla