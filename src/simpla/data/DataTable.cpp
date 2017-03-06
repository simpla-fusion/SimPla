//
// Created by salmon on 16-11-9.
//
#include "DataTable.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>
#include <string>
#include "DataBackend.h"
#include "DataBackendMemory.h"
#include "DataEntity.h"
namespace simpla {
namespace data {

KeyValue::KeyValue(unsigned long long int n, std::shared_ptr<DataEntity> const& p) : base_type(std::to_string(n), p) {}
KeyValue::KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p) : base_type(k, p) {}
KeyValue::KeyValue(KeyValue const& other) : base_type(other) {}
KeyValue::KeyValue(KeyValue&& other) : base_type(other) {}
KeyValue::~KeyValue() {}

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u) {
    auto res = std::make_shared<DataTable>();
    res->Set(u);
    return std::dynamic_pointer_cast<DataEntity>(res);
}
DataTable::DataTable(DataBackend* p)
    : DataEntity(), m_backend_(p != nullptr ? p : static_cast<DataBackend*>(new DataBackendMemory)){};
DataTable::DataTable(DataTable&& other) : m_backend_(other.m_backend_){};
DataTable::DataTable(DataTable const& other) : m_backend_(other.m_backend_){};
DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); }

std::type_info const& DataTable::backend_type() const { return m_backend_->type(); };
void DataTable::Parse(std::string const& str) { m_backend_->Parse(str); }
void DataTable::Open(std::string const& url, std::string const& status) { m_backend_->Open(url, status); }
void DataTable::Flush() { m_backend_->Flush(); }
void DataTable::Close() { m_backend_->Close(); }

std::ostream& DataTable::Print(std::ostream& os, int indent) const { return m_backend_->Print(os, indent); }
bool DataTable::empty() const { return m_backend_->empty(); }
void DataTable::clear() { return m_backend_->clear(); }
void DataTable::reset() { return m_backend_->reset(); }
bool DataTable::Erase(std::string const& k) { return m_backend_->Erase(k); }
DataTable* DataTable::CreateTable(std::string const& url) { m_backend_->CreateTable(url); }
DataEntity* DataTable::Set(std::string const& k, std::shared_ptr<DataEntity> const& v) { m_backend_->Set(k, v); }
DataEntity* DataTable::Get(std::string const& url) { return m_backend_->Get(url); }
DataEntity const* DataTable::Get(std::string const& url) const { return m_backend_->Get(url); }

}  // namespace data
}  // namespace simpla{namespace toolbox{