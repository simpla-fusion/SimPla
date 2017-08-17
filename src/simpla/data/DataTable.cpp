//
// Created by salmon on 16-11-9.
//

#include "simpla/SIMPLA_config.h"

#include <simpla/data/db/DataBaseMemory.h>
#include <iomanip>
#include <regex>
#include <string>
#include "DataBase.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/SingletonHolder.h"
namespace simpla {
namespace data {

DataTable::DataTable(std::shared_ptr<DataBase> const& db) : m_database_(db) { ASSERT(m_database_ != nullptr); };

DataTable::DataTable(std::string const& uri) : DataTable(DataBase::New(uri)){};

int DataTable::Flush() { return m_database_->Flush(); }

bool DataTable::isNull() const { return m_database_ == nullptr; }
size_type DataTable::Count() const { return 0; }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& path) { return m_database_->Get(path); };
std::shared_ptr<const DataEntity> DataTable::Get(std::string const& path) const { return m_database_->Get(path); };
int DataTable::Set(std::string const& uri, const std::shared_ptr<DataEntity>& p) {
    auto dst = std::dynamic_pointer_cast<DataTable>(Get(uri));
    auto src = std::dynamic_pointer_cast<DataTable>(p);

    if (dst != nullptr && src != nullptr) {
        dst->SetTable(*src);
    } else {
        m_database_->Set(uri, p);
    }
    return 1;
};
int DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& p) {
    auto dst = std::dynamic_pointer_cast<DataArray>(Get(uri));
    auto src = std::dynamic_pointer_cast<DataArray>(p);
    if (dst != nullptr && src != nullptr) {
        //        if (dst->value_type_info() == src->value_type_info() || dst->value_type_info() == typeid(void)) {
        //            dst->Add(p);
        //        } else {
        //            auto t_array = std::make_shared<DataArrayWrapper<>>();
        //            t_array->Add(Get(uri));
        //            t_array->Add(p);
        //            m_database_->Set(uri, t_array);
        //        }
    } else {
        auto t_array = DataArray::New();
        t_array->Add(p);
        m_database_->Set(uri, t_array);
    }

    return 1;
};

// std::ostream& DataTable::Serialize(std::ostream& os, int indent) const {
//    if (Count() == 0) { return os; };
//    os << "[";
//    Get(0)->Serialize(os, indent + 1);
//    for (size_type i = 1, ie = Count(); i < ie; ++i) {
//        os << ",";
//        Get(i)->Serialize(os, indent + 1);
//    }
//    os << "]";
//    return os;
//}
//******************************************************************************************************************
// void DataTable::Link(std::shared_ptr<DataEntity> const& other) { Link("", other); };
//
// DataTable& DataTable::Link(std::string const& uri, DataTable const& other) {
//    ASSERT(other.m_database_ != nullptr);
//    DataTable* res = nullptr;
//
//    if (uri.empty()) {
//        m_database_ = other.m_database_;
//        res = this;
//    } else {
//        m_database_->Set(uri, std::make_shared<DataTable>(other.m_database_));
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
        res = DataTable::New();
        Set(uri, res);
    } else if (std::dynamic_pointer_cast<DataTable>(res) == nullptr) {
        OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl;
    }
    return *std::dynamic_pointer_cast<DataTable>(res);
}

const DataTable& DataTable::GetTable(std::string const& uri) const {
    auto res = std::dynamic_pointer_cast<const DataTable>(Get(uri));
    if (res == nullptr) { OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl; }
    return *res;
}

int DataTable::Delete(std::string const& uri) { return m_database_->Delete(uri); };

void DataTable::SetTable(DataTable const& other) {
    other.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> v) { return Set(k, v); });
}
// void DataTable::SetValue(KeyValue const& item) { Set(item.first, item.second, true); }
//
// void DataTable::SetValue(std::initializer_list<KeyValue> const& other) {
//    for (auto const& item : other) { Set(item.first, item.second, true); }
//}

int DataTable::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return m_database_->Foreach(f);
}
//
// std::ostream& DataTable::Serialize(std::ostream& os, int indent) const {
//    os << "{";
//
//    m_database_->Foreach([&](std::string const& k, std::shared_ptr<DataEntity> const& v) {
//        os << std::endl
//           << std::setw(indent + 1) << " "
//           << "\"" << k << "\": ";
//        v->Serialize(os, indent + 1);
//        os << ",";
//    });
//
//    os << std::endl
//       << std::setw(indent) << " "
//       << "}";
//    return os;
//};
// std::istream& DataTable::Deserialize(std::istream& is) { return is; }

}  // namespace data
}  // namespace simpla