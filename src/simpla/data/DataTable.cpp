//
// Created by salmon on 16-11-9.
//

#include "simpla/SIMPLA_config.h"

#include <iomanip>
#include <regex>
#include <string>
#include "DataBackend.h"
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/SingletonHolder.h"
namespace simpla {
namespace data {
DataTable::DataTable() : m_backend_(DataBackend::Create("mem://", "")) { ASSERT(m_backend_ != nullptr); };
DataTable::DataTable(std::string const& uri, std::string const& param) : m_backend_(DataBackend::Create(uri, param)) {
    ASSERT(m_backend_ != nullptr);
};
DataTable::DataTable(std::shared_ptr<DataBackend> const& p) : m_backend_(p) { ASSERT(m_backend_ != nullptr); };
DataTable::DataTable(const DataTable& other) : m_backend_(other.m_backend_->Duplicate()) {
    ASSERT(m_backend_ != nullptr);
}
// DataTable::DataTable(std::initializer_list<KeyValue> const& l) : DataTable() {
//    for (auto const& item : l) { Set(item.first, item.second); }
//}

DataTable::DataTable(DataTable&& other) noexcept : m_backend_(other.m_backend_) {}

DataTable::~DataTable(){};
void DataTable::swap(DataTable& other) {
    std::swap(m_backend_, other.m_backend_);
    ASSERT(m_backend_ != nullptr);
};

//******************************************************************************************************************

void DataTable::Flush() { m_backend_->Flush(); }

std::shared_ptr<DataEntity> DataTable::Duplicate() const {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(m_backend_->Duplicate()));
}

bool DataTable::isNull() const { return m_backend_ == nullptr; }
size_type DataTable::size() const { return m_backend_->size(); }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) const { return m_backend_->Get(path); };
void DataTable::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    bool success = false;
    bool overwrite = true;
    auto res = Get(uri);
    if (res != nullptr && !overwrite) { return; }

    if (dynamic_cast<DataTable const*>(v.get()) != nullptr) {
        if (!overwrite && res != nullptr && dynamic_cast<DataTable const*>(res.get()) == nullptr) {
            success = false;
        } else if (dynamic_cast<DataTable const*>(res.get()) == nullptr) {
            res = std::make_shared<DataTable>();
        }
        auto dest_table = std::dynamic_pointer_cast<DataTable>(res);
        auto src_table = std::dynamic_pointer_cast<DataTable>(v);
        src_table->Foreach(
            [&](std::string const& k, std::shared_ptr<DataEntity> const& tv) { dest_table->Set(k, tv); });
        success = true;
    } else if (dynamic_cast<DataEntityWrapper<void*> const*>(v.get()) != nullptr) {
        auto dest_array = std::make_shared<DataEntityWrapper<void*>>();
        auto src_array = std::dynamic_pointer_cast<DataArray>(v);
        for (size_type i = 0, ie = src_array->size(); i < ie; ++i) { dest_array->Add(src_array->Get(i)); }
        res = dest_array;
        success = true;
    } else if (res == nullptr || overwrite) {
        res = v;
        success = true;
    }
    if (success) { m_backend_->Set(uri, res); }
};
void DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) { m_backend_->Add(uri, v); };
//******************************************************************************************************************
void DataTable::Link(std::shared_ptr<DataEntity> const& other) { Link("", other); };

DataTable& DataTable::Link(std::string const& uri, DataTable const& other) {
    ASSERT(other.m_backend_ != nullptr);
    DataTable* res = nullptr;

    if (uri.empty()) {
        m_backend_ = other.m_backend_;
        res = this;
    } else {
        m_backend_->Set(uri, std::make_shared<DataTable>(other.m_backend_), true);
        res = dynamic_cast<DataTable*>(Get(uri).get());
    }
    return *res;
}

DataTable& DataTable::Link(std::string const& uri, std::shared_ptr<DataEntity> const& other) {
    if (dynamic_cast<DataTable const*>(other.get()) == nullptr) {
        RUNTIME_ERROR << "link array or entity to table" << std::endl;
    }
    return Link(uri, *std::dynamic_pointer_cast<DataTable>(other));
}

void DataTable::Set(std::string const& uri, DataEntity const& p) { Set(uri, p.Duplicate()); };
void DataTable::Add(std::string const& uri, DataEntity const& p) { Add(uri, p.Duplicate()); };

std::shared_ptr<DataTable> DataTable::GetTable(std::string const& uri) const {
    return std::dynamic_pointer_cast<DataTable>(Get(uri));
}

size_type DataTable::Delete(std::string const& uri) { return m_backend_->Delete(uri); };

void DataTable::Set(DataTable const& other) {
    other.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
}
// void DataTable::SetValue(KeyValue const& item) { Set(item.first, item.second, true); }
//
// void DataTable::SetValue(std::initializer_list<KeyValue> const& other) {
//    for (auto const& item : other) { Set(item.first, item.second, true); }
//}

size_type DataTable::Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return m_backend_->Foreach(f);
}

std::ostream& DataTable::Serialize(std::ostream& os, int indent) const {
    os << "{";

    m_backend_->Foreach([&](std::string const& k, std::shared_ptr<DataEntity> const& v) {
        os << std::endl
           << std::setw(indent + 1) << " "
           << "\"" << k << "\": ";
        v->Serialize(os, indent + 1);
        os << ",";
    });

    os << std::endl
       << std::setw(indent) << " "
       << "}";
    return os;
};
std::istream& DataTable::Deserialize(std::istream& is) { return is; }

}  // namespace data
}  // namespace simpla