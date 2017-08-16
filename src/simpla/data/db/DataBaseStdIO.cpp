//
// Created by salmon on 17-8-16.
//

#include "DataBaseStdIO.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseStdIO, stdio);

struct DataBaseStdIO::pimpl_s {};
DataBaseStdIO::DataBaseStdIO() : m_pimpl_(new pimpl_s) {}
DataBaseStdIO::~DataBaseStdIO() { delete m_pimpl_; }
void DataBaseStdIO::SetStream(std::ostream&) {}
void DataBaseStdIO::SetStream(std::istream&) {}
int DataBaseStdIO::Connect(std::string const& authority, std::string const& path, std::string const& query,
                           std::string const& fragment) {
    return SP_SUCCESS;
}

int DataBaseStdIO::Disconnect() { return SP_SUCCESS; }
int DataBaseStdIO::Flush() { return SP_SUCCESS; }
bool DataBaseStdIO::isNull(std::string const& uri) const { return true; }
size_type DataBaseStdIO::Count(std::string const& url) const { return 0; }
std::shared_ptr<DataEntity> DataBaseStdIO::Get(std::string const& URI) const { return nullptr; }
int DataBaseStdIO::Set(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseStdIO::Add(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseStdIO::Delete(std::string const& URI) { return 0; }
int DataBaseStdIO::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return 0;
}
}
}