//
// Created by salmon on 17-8-24.
//

#include "DataNodeIMAS.h"
namespace simpla {
namespace data {

REGISTER_CREATOR(DataNodeIMAS, imas);
struct DataNodeIMAS::pimpl_s {};
DataNodeIMAS::DataNodeIMAS() : m_pimpl_(new pimpl_s) {}
DataNodeIMAS::DataNodeIMAS(pimpl_s* pimpl) : m_pimpl_(pimpl) {}
DataNodeIMAS::~DataNodeIMAS() { delete m_pimpl_; }

std::shared_ptr<DataNode> DataNodeIMAS::Duplicate() const { return New(); }

size_type DataNodeIMAS::size() const { return 0; }
DataNode::eNodeType DataNodeIMAS::type() const { return DN_NULL; }

int DataNodeIMAS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    return 0;
}
int DataNodeIMAS::Disconnect() { return 0; }
bool DataNodeIMAS::isValid() const { return false; }
int DataNodeIMAS::Flush() { return 0; }

std::shared_ptr<DataNode> DataNodeIMAS::Root() const { return New(); }
std::shared_ptr<DataNode> DataNodeIMAS::Parent() const { return New(); }

size_type DataNodeIMAS::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) {
    return 0;
}
size_type DataNodeIMAS::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    return 0;
}

std::shared_ptr<DataNode> DataNodeIMAS::GetNode(std::string const& uri, int flag) { return New(); }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(std::string const& uri, int flag) const { return New(); }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(index_type s, int flag) { return New(); }
std::shared_ptr<DataNode> DataNodeIMAS::GetNode(index_type s, int flag) const { return New(); }
size_type DataNodeIMAS::DeleteNode(std::string const& uri, int flag) { return 0; }
void DataNodeIMAS::Clear() {}
std::shared_ptr<DataEntity> DataNodeIMAS::GetEntity() const { return DataEntity::New(); }
size_type DataNodeIMAS::SetEntity(std::shared_ptr<DataEntity> const& v) { return 0; }
size_type DataNodeIMAS::AddEntity(std::shared_ptr<DataEntity> const& v) { return 0; }

}  // { namespace data {
}  // namespace simpla