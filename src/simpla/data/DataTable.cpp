//
// Created by salmon on 16-11-9.
//
#include "DataTable.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>
#include <string>
#include "DataBackend.h"
#include "DataBackendFactroy.h"
#include "DataBackendMemory.h"
#include "DataEntity.h"
namespace simpla {
namespace data {
DataTable::DataTable(std::shared_ptr<DataBackend> const& p)
    : m_backend_((p != nullptr) ? p : CreateDataBackendFromFile()){};
DataTable::DataTable(std::string const& url, std::string const& status)
    : m_backend_(CreateDataBackendFromFile(url, status)){};
DataTable::DataTable(const DataTable& other) : m_backend_(other.m_backend_) {}
DataTable::DataTable(DataTable&& other) : m_backend_(other.m_backend_) { other.m_backend_.reset(); }
DataTable::~DataTable(){};
std::ostream& DataTable::Print(std::ostream& os, int indent) const { return m_backend_->Print(os, indent); }
void DataTable::Open(std::string const& url, std::string const& status) {
    Close();
    m_backend_ = CreateDataBackendFromFile(url, status);
}
void DataTable::Parse(std::string const& str) { m_backend_->Parse(str); }
void DataTable::Flush() { m_backend_->Flush(); }
void DataTable::Close() { m_backend_->Close(); }
std::shared_ptr<DataTable> DataTable::Copy() const { return std::make_shared<DataTable>(*this); }
bool DataTable::empty() const { return m_backend_ == nullptr || m_backend_->empty(); }
size_type DataTable::count() const { return m_backend_ == nullptr ? 0 : m_backend_->count(); }
void DataTable::Reset() { m_backend_->Reset(); }
void DataTable::Clear() { m_backend_->Clear(); }

std::shared_ptr<DataEntity> DataTable::Get(std::string const& key) const { return m_backend_->Get(key); };
bool DataTable::Set(DataTable const& other) { return false; }
bool DataTable::Set(std::string const& key, std::shared_ptr<DataEntity> const& v) { return m_backend_->Set(key, v); };
bool DataTable::Add(std::string const& key, std::shared_ptr<DataEntity> const& v) { return m_backend_->Add(key, v); };
size_type DataTable::Delete(std::string const& key) { return m_backend_->Delete(key); };

}  // namespace data
}  // namespace simpla