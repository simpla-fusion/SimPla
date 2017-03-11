//
// Created by salmon on 16-11-9.
//
#include "DataTable.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/toolbox/Log.h>
#include <iomanip>
#include <regex>
#include <string>
#include "DataBackend.h"
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "KeyValue.h"
namespace simpla {
namespace data {
DataTable::DataTable() : m_backend_(new DataBackendMemory), m_base_uri_("mem://"){};
DataTable::DataTable(std::string const& uri) : m_backend_(nullptr), m_base_uri_(uri) {
    static std::regex uri_regex(R"(^(([^:\/?#]+):)?(//(.*)))", std::regex::extended | std::regex::optimize);
    std::smatch uri_match_result;

    if (!std::regex_match(m_base_uri_, uri_match_result, uri_regex)) {
        RUNTIME_ERROR << " illegal uri! [" << uri << "]" << std::endl;
    }

    std::string scheme = uri_match_result[2].str();
    std::string path = uri_match_result[4].str();

    std::unique_ptr<DataBackend>(DataBackend::Create(scheme)).swap(m_backend_);
    m_backend_->Connect(path);
    //    auto res = this->Get(path);
    //    if (!res->isTable()) { RUNTIME_ERROR << "uri does not point  to a table! [ " << uri << " ]" << std::endl; }
    //    res->cast_as<DataTable>().swap(*this);
};
DataTable::DataTable(DataBackend* p) : m_backend_(p){};
DataTable::DataTable(std::unique_ptr<DataBackend>&& p) : m_backend_(std::move(p)){};
DataTable::DataTable(const DataTable& other) : m_backend_(std::move(other.m_backend_->Clone())) { Set(other); }
DataTable::DataTable(DataTable&& other) : m_backend_(std::move(other.m_backend_)) {}
DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); };
void DataTable::Flush() { m_backend_->Flush(); }

std::shared_ptr<DataTable> DataTable::Create(std::string const& scheme) {
    return std::make_shared<DataTable>(DataBackend::Create(scheme));
}
std::shared_ptr<DataEntity> DataTable::Clone() const {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(m_backend_->Clone()));
}

bool DataTable::isNull() const { return m_backend_ == nullptr; }
size_type DataTable::size() const { return m_backend_->size(); }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) const { return m_backend_->Get(path); };
std::shared_ptr<DataEntity> DataTable::Get(id_type key) const { return m_backend_->Get(key); };
bool DataTable::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) { return m_backend_->Set(uri, v); };
bool DataTable::Set(id_type key, std::shared_ptr<DataEntity> const& v) { return m_backend_->Set(key, v); };
bool DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) { return m_backend_->Add(uri, v); };
bool DataTable::Add(id_type key, std::shared_ptr<DataEntity> const& v) { return m_backend_->Add(key, v); };
size_type DataTable::Delete(std::string const& uri) { return m_backend_->Delete(uri); };
size_type DataTable::Delete(id_type key) { return m_backend_->Delete(key); };

std::shared_ptr<DataTable> DataTable::AddTable(std::string const& uri) {
    std::shared_ptr<DataEntity> res = Get(uri);
    if (res == nullptr && !res->isTable()) {
        res = std::make_shared<DataTable>(m_backend_->Clone());
        Set(uri, res);
    }
    return std::dynamic_pointer_cast<DataTable>(res);
};

bool DataTable::Set(DataTable const& other) {
    other.Accept([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
    return true;
}
bool DataTable::Set(std::initializer_list<KeyValue> const& other) {
    for (auto const& item : other) { Set(item.first, item.second); }
    return true;
}

size_type DataTable::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return m_backend_->Accept(f);
}
size_type DataTable::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const& f) const {
    return m_backend_->Accept(f);
}
std::ostream& DataTable::Print(std::ostream& os, int indent) const {
    os << "{";
    m_backend_->Accept([&](std::string const& k, std::shared_ptr<DataEntity> const& v) {
        os << std::endl
           << std::setw(indent + 1) << " "
           << "\"" << k << "\": ";
        v->Print(os, indent + 1);
        os << ",";
    });

    os << std::endl
       << std::setw(indent) << " "
       << " }";
    return os;
};

}  // namespace data
}  // namespace simpla