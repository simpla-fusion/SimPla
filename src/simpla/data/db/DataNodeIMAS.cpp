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

std::shared_ptr<DataNode> DataNodeIMAS::Root() const { return New(); }
std::shared_ptr<DataNode> DataNodeIMAS::Parent() const { return New(); }

size_type DataNodeIMAS::Foreach(
    std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& fun) const {
    return 0;
}

size_type DataNodeIMAS::Set(std::string const &url, std::shared_ptr<DataEntity> const &v) { return 0; }
size_type DataNodeIMAS::Add(std::string const& url, std::shared_ptr<DataEntity> const& v) { return 0; }
size_type DataNodeIMAS::Delete(std::string const& uri) { return 0; }
std::shared_ptr<const DataNode> DataNodeIMAS::Get(std::string const& uri) const { return New(); }

std::shared_ptr<DataEntity> DataNodeIMAS::GetEntity() const { return DataEntity::New(); }

}  // { namespace data {
}  // namespace simpla