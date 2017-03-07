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
KeyValue::KeyValue(KeyValue const& other) : base_type(other) {}
KeyValue::KeyValue(KeyValue&& other) : base_type(other) {}
KeyValue::~KeyValue() {}

DataEntity make_data_entity(std::initializer_list<KeyValue> const& u) {
    DataEntity res;
    for (auto const& item : u) { res[item.first] = item.second; }
    return std::move(res);
}
DataTable::DataTable(DataBackend* p) : m_backend_(p){};
DataTable::~DataTable() {
    if (m_backend_ != nullptr) { delete m_backend_; }
};
DataTable::DataTable(std::string const& url, std::string const& status) { Open(url, status); };

DataTable::DataTable(DataTable&& other) : m_backend_(other.m_backend_) { other.m_backend_ = nullptr; };
DataTable::DataTable(DataTable const& other) : m_backend_(other.m_backend_->Copy()){};

void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); }
void DataTable::Update() { m_backend_ = (m_backend_ == nullptr) ? new DataBackendMemory : m_backend_; };
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
DataHolderBase* DataTable::Copy() const { return new DataTable(*this); }
bool DataTable::empty() const { return m_backend_ == nullptr || m_backend_->empty(); }
void DataTable::Reset() {
    if (m_backend_ != nullptr) { delete m_backend_; }
    m_backend_ = nullptr;
}
void DataTable::Clear() {
    Update();
    m_backend_->Clear();
}

bool DataTable::Erase(std::string const& k) { return (m_backend_ == nullptr) ? false : m_backend_->Erase(k); }

std::pair<DataEntity*, bool> DataTable::Insert(std::string const& k) {
    Update();
    return m_backend_->Insert(k, DataEntity(), true);
};

std::pair<DataEntity*, bool> DataTable::Insert(std::string const& k, DataEntity const& v, bool assign_is_exists) {
    Update();
    return m_backend_->Insert(k, v, assign_is_exists);
};

DataEntity* DataTable::Find(std::string const& url) const { return m_backend_->Find(url); }
DataEntity& DataTable::operator[](std::string const& url) { return *Insert(url).first; };
DataEntity const& DataTable::operator[](std::string const& url) const {
    auto p = Find(url);
    ASSERT(p != nullptr);
    return *p;
};
}  // namespace data
}  // namespace simpla{namespace toolbox{