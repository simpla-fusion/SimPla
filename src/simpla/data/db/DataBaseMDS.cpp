//
// Created by salmon on 17-8-15.
//

#include "DataBaseMDS.h"
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
bool DataBaseMDS::isNull(std::string const& uri) const { return true; }
size_type DataBaseMDS::Count(std::string const& url) const { return 0; }
std::shared_ptr<DataEntity> DataBaseMDS::Get(std::string const& URI) const { return nullptr; }
int DataBaseMDS::Set(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseMDS::Add(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseMDS::Delete(std::string const& URI) { return 0; }
int DataBaseMDS::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return 0;
}
}  // namespace data {
}  // namespace simpla {
