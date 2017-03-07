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

KeyValue::KeyValue(unsigned long long int n, DataEntity const& p) : base_type(std::to_string(n), p) {}
KeyValue::KeyValue(std::string const& k, DataEntity const& p) : base_type(k, p) {}
KeyValue::KeyValue(std::string const& k, DataEntity&& p) : base_type(k, std::forward<DataEntity>(p)) {}
KeyValue::KeyValue(KeyValue const& other) : base_type(other) {}
KeyValue::KeyValue(KeyValue&& other) : base_type(other) {}
KeyValue::~KeyValue() {}

DataEntity make_data_entity(std::initializer_list<KeyValue> const& u) {
    auto res = new DataTable;
    for (auto const& item : u) { res->Put(item.first, item.second); }
    return std::move(DataEntity{res});
}
DataTable::DataTable(DataBackend* p) : m_backend_(p){};
DataTable::~DataTable() {
    if (m_backend_ != nullptr) { delete m_backend_; }
};
DataTable::DataTable(std::string const& url, std::string const& status) { Open(url, status); };
DataTable::DataTable(DataTable&& other) : m_backend_(other.m_backend_) { other.m_backend_ = nullptr; };
DataTable::DataTable(DataTable const& other)
    : m_backend_(other.m_backend_ == nullptr ? nullptr : other.m_backend_->Copy()){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); }
bool DataTable::Update() {
    m_backend_ = (m_backend_ == nullptr) ? static_cast<DataBackend*>(new DataBackendMemory) : m_backend_;
    return true;
};
DataTable& DataTable::operator=(DataTable const& other) {
    DataTable(other).swap(*this);
    return *this;
}

void DataTable::Parse(std::string const& str) {
    Update();
    m_backend_->Parse(str);
}
void DataTable::Open(std::string const& url, std::string const& status) {
    if (url == "") { return; }
    if (m_backend_ == nullptr) {
        m_backend_ = CreateDataBackendFromFile(url);
    } else {
        m_backend_->Open(url, status);
    }
}
void DataTable::Flush() {
    if (m_backend_ != nullptr) { m_backend_->Flush(); }
}
void DataTable::Close() {
    if (m_backend_ != nullptr) { m_backend_->Close(); }
}

std::ostream& DataTable::Print(std::ostream& os, int indent) const {
    ASSERT(m_backend_ != nullptr);
    return m_backend_->Print(os, indent);
}
DataTable* DataTable::Copy() const { return new DataTable(*this); }
bool DataTable::empty() const { return m_backend_ == nullptr || m_backend_->empty(); }
void DataTable::Reset() {
    if (m_backend_ != nullptr) { delete m_backend_; }
    m_backend_ = nullptr;
}
void DataTable::Clear() {
    if (Update()) m_backend_->Clear();
}

DataEntity DataTable::Get(std::string const& uri) { return std::move(Update() ? m_backend_->Get(uri) : DataEntity{}); }
bool DataTable::Put(std::string const& uri, DataEntity&& v) {
    return Update() ? m_backend_->Put(uri, std::forward<DataEntity>(v)) : false;
}
bool DataTable::Post(std::string const& uri, DataEntity&& v) {
    return Update() ? m_backend_->Post(uri, std::forward<DataEntity>(v)) : false;
}
size_type DataTable::Delete(std::string const& uri) { return (m_backend_ == nullptr) ? 0 : m_backend_->Delete(uri); }
size_type DataTable::Count(std::string const& uri) const {
    return (m_backend_ == nullptr) ? 0 : m_backend_->Count(uri);
}

}  // namespace data
}  // namespace simpla