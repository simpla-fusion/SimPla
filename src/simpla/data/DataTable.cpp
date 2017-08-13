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
DataTable::DataTable(std::string const& uri, std::string const& param) : m_backend_(DataBackend::Create(uri, param)){};
DataTable::DataTable(std::shared_ptr<DataBackend> const& p) : m_backend_(p) { ASSERT(m_backend_ != nullptr); };
DataTable::DataTable(const DataTable& other) : m_backend_(other.m_backend_->Duplicate()) {}
DataTable::DataTable(DataTable&& other) noexcept : m_backend_(std::move(other.m_backend_)) {}

void DataTable::swap(DataTable& other) {
    std::swap(m_backend_, other.m_backend_);
    ASSERT(m_backend_ != nullptr);
};

//******************************************************************************************************************

void DataTable::Flush() { m_backend_->Flush(); }

bool DataTable::isNull() const { return m_backend_ == nullptr; }
size_type DataTable::size() const { return m_backend_->size(); }

std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) { return m_backend_->Get(path); };
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) const { return m_backend_->Get(path); };
void DataTable::Set(std::string const& uri, const std::shared_ptr<DataEntity>& p) {
    auto dst = std::dynamic_pointer_cast<DataTable>(Get(uri));
    auto src = std::dynamic_pointer_cast<DataTable>(p);

    if (dst != nullptr && src != nullptr) {
        dst->SetTable(*src);
    } else {
        m_backend_->Set(uri, p);
    }
};
void DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& p) {
    auto dst = std::dynamic_pointer_cast<DataArray>(Get(uri));
    auto src = std::dynamic_pointer_cast<DataArray>(p);
    if (dst != nullptr && src != nullptr) {
        if (dst->value_type_info() == src->value_type_info() || dst->value_type_info() == typeid(void)) {
            dst->Add(p);
        } else {
            auto t_array = std::make_shared<DataArrayWrapper<>>();
            t_array->Add(Get(uri));
            t_array->Add(p);
            m_backend_->Set(uri, t_array);
        }
    } else {
        auto t_array = std::make_shared<DataArrayWrapper<>>();
        t_array->Add(p);
        m_backend_->Set(uri, t_array);
    }
};
//******************************************************************************************************************
// void DataTable::Link(std::shared_ptr<DataEntity> const& other) { Link("", other); };
//
// DataTable& DataTable::Link(std::string const& uri, DataTable const& other) {
//    ASSERT(other.m_backend_ != nullptr);
//    DataTable* res = nullptr;
//
//    if (uri.empty()) {
//        m_backend_ = other.m_backend_;
//        res = this;
//    } else {
//        m_backend_->Set(uri, std::make_shared<DataTable>(other.m_backend_));
//        res = dynamic_cast<DataTable*>(Get(uri).get());
//    }
//    return *res;
//}
//
// DataTable& DataTable::Link(std::string const& uri, std::shared_ptr<DataEntity> const& other) {
//    if (dynamic_cast<DataTable const*>(other.get()) == nullptr) {
//        RUNTIME_ERROR << "link array or entity to table" << std::endl;
//    }
//    return Link(uri, *std::dynamic_pointer_cast<DataTable>(other));
//}
// void DataTable::Set(std::string const& uri, const std::shared_ptr<DataEntity>& p) { Set(uri, p.Duplicate()); };
// void DataTable::Add(std::string const& uri, const std::shared_ptr<DataEntity>& p) { Add(uri, p.Duplicate()); };

DataTable& DataTable::GetTable(std::string const& uri) {
    auto res = Get(uri);
    if (res == nullptr) {
        res = std::make_shared<DataTable>();
        Set(uri, res);
    } else if (std::dynamic_pointer_cast<DataTable>(res) == nullptr) {
        OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl;
    }
    return *std::dynamic_pointer_cast<DataTable>(res);
}

const DataTable& DataTable::GetTable(std::string const& uri) const {
    auto res = std::dynamic_pointer_cast<DataTable>(Get(uri));
    if (res == nullptr) { OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl; }
    return *res;
}

size_type DataTable::Delete(std::string const& uri) { return m_backend_->Delete(uri); };

void DataTable::SetTable(DataTable const& other) {
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