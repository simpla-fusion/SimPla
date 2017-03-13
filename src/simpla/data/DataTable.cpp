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
DataTable::DataTable() : m_backend_(GLOBAL_DATA_BACKEND_FACTORY.Create("mem://")){};
DataTable::DataTable(std::string const& uri, std::string const& param)
    : m_backend_(GLOBAL_DATA_BACKEND_FACTORY.Create(uri, param)) {
    ;
};
DataTable::DataTable(std::shared_ptr<DataBackend> const& p) : m_backend_(p){};
DataTable::DataTable(const DataTable& other) : m_backend_(std::move(other.m_backend_->Duplicate())) {}
DataTable::DataTable(DataTable&& other) : m_backend_(std::move(other.m_backend_)) {}
DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); };
void DataTable::Flush() { m_backend_->Flush(); }

std::shared_ptr<DataEntity> DataTable::Clone() const {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(m_backend_->Duplicate()));
}

bool DataTable::isNull() const { return m_backend_ == nullptr; }
size_type DataTable::size() const { return m_backend_->size(); }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) const { return m_backend_->Get(path); };
void DataTable::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) { m_backend_->Set(uri, v); };
void DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) { m_backend_->Add(uri, v); };
size_type DataTable::Delete(std::string const& uri) { return m_backend_->Delete(uri); };

void DataTable::SetValue(DataTable const& other) {
    other.Accept([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
}
void DataTable::SetValue(std::initializer_list<KeyValue> const& other) {
    for (auto const& item : other) { Set(item.first, item.second); }
}

size_type DataTable::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
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