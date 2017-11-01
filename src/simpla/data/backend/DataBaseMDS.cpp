//
// Created by salmon on 17-8-15.
//

#include "DataBaseMDS.h"

#include "../DataNode.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseMDS, mds);

struct DataBaseMDS::pimpl_s {};
DataBaseMDS::DataBaseMDS() : m_pimpl_(new pimpl_s) {}
DataBaseMDS::~DataBaseMDS() { delete m_pimpl_; }

int DataBaseMDS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    return SP_SUCCESS;
}
int DataBaseMDS::Disconnect() { return SP_SUCCESS; }
int DataBaseMDS::Flush() { return SP_SUCCESS; }
bool DataBaseMDS::isNull() const { return true; }

std::shared_ptr<DataNode> DataBaseMDS::Root() { return DataNode::New(); }

}  // namespace data {
}  // namespace simpla {
