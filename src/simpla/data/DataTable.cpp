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
#include "DataTableFactroy.h"

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
DataTable::DataTable(std::shared_ptr<DataBackend> p) : DataEntity(), m_backend_(p){};
DataTable::DataTable(std::string const& url, std::string const& status) : DataEntity() {
    if (url != "") { Open(url, status); }
};

DataTable::DataTable(DataTable&& other) : m_backend_(other.m_backend_){};
DataTable::DataTable(DataTable const& other) : m_backend_(other.m_backend_){};
DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); }
void DataTable::Update() {
    if (m_backend_ == nullptr) {
        m_backend_ = std::dynamic_pointer_cast<DataBackend>(std::make_shared<DataBackendMemory>());
    }
};
std::type_info const& DataTable::backend_type() const {
    ASSERT(m_backend_ != nullptr);
    return m_backend_->type();
};
void DataTable::Parse(std::string const& str) {
    Update();
    m_backend_->Parse(str);
}
void DataTable::Open(std::string const& url, std::string const& status) {
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
bool DataTable::empty() const { return m_backend_ == nullptr || m_backend_->empty(); }
void DataTable::Reset() { m_backend_.reset(); }
void DataTable::Clear() {
    Update();
    m_backend_->Clear();
}

bool DataTable::Erase(std::string const& k) { return (m_backend_ == nullptr) ? false : m_backend_->Erase(k); }

std::shared_ptr<DataTable> DataTable::CreateTable(std::string const& url) {
    Update();
    m_backend_->CreateTable(url);
}
std::shared_ptr<DataEntity> DataTable::Set(std::string const& k, std::shared_ptr<DataEntity> const& v) {
    Update();
    return m_backend_->Set(k, v);
}
std::shared_ptr<DataEntity> DataTable::Get(std::string const& url) {
    Update();
    return m_backend_->Get(url);
}
std::shared_ptr<DataEntity> DataTable::Get(std::string const& url) const {
    ASSERT(m_backend_ != nullptr);
    return m_backend_->Get(url);
}

}  // namespace data
}  // namespace simpla{namespace toolbox{