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
    : m_backend_(GLOBAL_DATA_BACKEND_FACTORY.Create(uri, param)){};
DataTable::DataTable(std::shared_ptr<DataBackend> const& p) : m_backend_(p){};
DataTable::DataTable(const DataTable& other) : m_backend_(std::move(other.m_backend_->Duplicate())) {}
DataTable::DataTable(DataTable&& other) : m_backend_(std::move(other.m_backend_)) {}
DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) { std::swap(m_backend_, other.m_backend_); };

//******************************************************************************************************************

void DataTable::Flush() { m_backend_->Flush(); }

std::shared_ptr<DataEntity> DataTable::Duplicate() const {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(m_backend_->Duplicate()));
}

bool DataTable::isNull() const { return m_backend_ == nullptr; }
size_type DataTable::size() const { return m_backend_->size(); }
void DataTable::Link(DataTable const& other) { m_backend_ = other.m_backend_; }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) const { return m_backend_->Get(path); };
std::shared_ptr<DataEntity> DataTable::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v,
                                           bool overwrite) {
    return m_backend_->Set(uri, v, overwrite);
};
std::shared_ptr<DataEntity> DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    return m_backend_->Add(uri, v);
};
//******************************************************************************************************************

void DataTable::Link(std::shared_ptr<DataEntity> const& other) {
    if (!other->isTable()) { RUNTIME_ERROR << "link array or entity to table" << std::endl; }
    Link(other->cast_as<DataTable>());
}

std::shared_ptr<DataEntity> DataTable::Set(std::string const& uri, DataEntity const& p, bool overwrite) {
    return Set(uri, p.Duplicate(), overwrite);
};
std::shared_ptr<DataEntity> DataTable::Add(std::string const& uri, DataEntity const& p) {
    return Add(uri, p.Duplicate());
};

DataTable const& DataTable::GetTable(std::string const& uri) const {
    auto p = Get(uri);
    if (p == nullptr || !p->isTable()) { RUNTIME_ERROR << uri << " is not a table" << std::endl; }
    return p->cast_as<DataTable>();
}
DataTable& DataTable::GetTable(std::string const& uri) {
    auto p = Set(uri + "/", nullptr, false);
    if (p == nullptr || !p->isTable()) { RUNTIME_ERROR << uri << " is not a table" << std::endl; }
    return p->cast_as<DataTable>();
}

size_type DataTable::Delete(std::string const& uri) { return m_backend_->Delete(uri); };

void DataTable::Set(DataTable const& other, bool overwrite) {
    other.ForEach([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v, overwrite); });
}
void DataTable::SetValue(std::initializer_list<KeyValue> const& other) {
    for (auto const& item : other) { Set(item.first, item.second); }
}

size_type DataTable::ForEach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return m_backend_->ForEach(f);
}

std::ostream& DataTable::Print(std::ostream& os, int indent) const {
    os << "{";

    m_backend_->ForEach([&](std::string const& k, std::shared_ptr<DataEntity> const& v) {
        os << std::endl
           << std::setw(indent + 1) << " "
           << "\"" << k << "\": ";
        v->Print(os, indent + 1);
        os << ",";
    });

    os << std::endl
       << std::setw(indent) << " "
       << "}";
    return os;
};

}  // namespace data
}  // namespace simpla