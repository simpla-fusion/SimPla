//
// Created by salmon on 17-8-13.
//
#include "DataBackendXDMF.h"

#include <Xdmf.hpp>

namespace simpla {
namespace data {
struct DataBackendXDMF::pimpl_s {
    void* m_file_ = nullptr;
};
DataBackendXDMF::DataBackendXDMF() : m_pimpl_(new pimpl_s) {}
DataBackendXDMF::~DataBackendXDMF() { delete m_pimpl_; }

DataBackendXDMF::DataBackendXDMF(std::string const& uri, std::string const& status) {}

bool DataBackendXDMF::isNull() const { return m_pimpl_->m_file_ == nullptr; }

void DataBackendXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                              std::string const& fragment) {}

void DataBackendXDMF::Disconnect() {}

std::shared_ptr<DataBackend> DataBackendXDMF::Duplicate() const {
    auto res = std::dynamic_pointer_cast<DataBackendXDMF>(CreateNew());
    res->m_pimpl_->m_file_ = this->m_pimpl_->m_file_;
    return res;
}
std::shared_ptr<DataBackend> DataBackendXDMF::CreateNew() const { return std::make_shared<DataBackendXDMF>(); }

void DataBackendXDMF::Flush() {}

std::shared_ptr<DataEntity> DataBackendXDMF::Get(std::string const& URI) const { return nullptr; }
void DataBackendXDMF::Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity) {}
void DataBackendXDMF::Add(std::string const& URI, const std::shared_ptr<DataEntity>& entity) {}
size_type DataBackendXDMF::Delete(std::string const& URI) { return 0; }
size_type DataBackendXDMF::size() const { return 0; }
size_type DataBackendXDMF::Foreach(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
    return 0;
}

}  // namespace data{
}  // namespace simpla