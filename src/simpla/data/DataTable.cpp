//
// Created by salmon on 16-11-9.
//

#include "simpla/SIMPLA_config.h"

#include <iomanip>
#include <regex>
#include <string>

#include "simpla/utilities/Log.h"
#include "simpla/utilities/SingletonHolder.h"

#include "DataArray.h"
#include "DataBase.h"
#include "DataEntity.h"
#include "DataEntry.h"
#include "DataTable.h"
namespace simpla {
namespace data {
struct DataTable::pimpl_s {
    std::shared_ptr<DataEntry> m_entry_;
    std::map<std::string, std::shared_ptr<DataEntity>> m_data_;
};
DataTable::DataTable() : m_pimpl_(new pimpl_s){};
DataTable::~DataTable() { delete m_pimpl_; };

bool DataTable::isNull() const { return m_pimpl_->m_entry_ == nullptr; }
size_type DataTable::Count() const { return m_pimpl_->m_data_.size(); }
std::shared_ptr<DataEntity> DataTable::Get(std::string const& uri) { return m_pimpl_->m_data_[uri]; };
std::shared_ptr<DataEntity> DataTable::Get(std::string const& uri) const { return m_pimpl_->m_data_[uri]; };
int DataTable::Set(std::string const& uri, const std::shared_ptr<DataEntity>& src) {
    m_pimpl_->m_data_[uri] = src;
    return 1;
};
int DataTable::Set(std::shared_ptr<DataTable> const& other) { return m_pimpl_->m_entry_->Set(other); }
int DataTable::Delete(std::string const& uri) { return static_cast<int>(m_pimpl_->m_data_.erase(uri)); }

int DataTable::Add(std::string const& uri, std::shared_ptr<DataEntity> const& src) {
    auto p = Get(uri);
    int count = 0;

    if (p == nullptr && src == nullptr) {
        p = DataArrayT<void>::New();
        count = 1;
    } else if (p == nullptr) {
        p = src;
        count = 1;
    } else if (src == nullptr) {  // DO NOTHING
    } else if (auto dst = std::dynamic_pointer_cast<DataArrayT<void>>(p)) {
        count = dst->Add(src);
    } else if (std::dynamic_pointer_cast<DataArray>(p) != nullptr &&
               std::dynamic_pointer_cast<DataLight>(src) != nullptr &&
               p->value_type_info() == src->value_type_info()) {  // dst is DataArrayT<V>
        count = std::dynamic_pointer_cast<DataArray>(p)->Add(src);
    } else {
        auto res = DataArrayT<void>::New();
        res->Add(p);
        count = res->Add(src);
        p = res;
    }
    return count;
};
//
// std::shared_ptr<DataTable> DataTable::GetTable(std::string const& uri) {
//    auto res = Get(uri);
//    if (res == nullptr) {
//        res = DataTable::New();
//        Set(uri, res);
//    } else if (std::dynamic_pointer_cast<DataTable>(res) == nullptr) {
//        OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl;
//    }
//    return *std::dynamic_pointer_cast<DataTable>(res);
//}
//
// std::shared_ptr<const DataTable> DataTable::GetTable(std::string const& uri) const {
//    auto res = std::dynamic_pointer_cast<const DataTable>(Get(uri));
//    if (res == nullptr) { OUT_OF_RANGE << "[" << uri << "] is not a table!" << std::endl; }
//    return *res;
//}
//
// int DataTable::Delete(std::string const& uri) { return m_database_->Delete(uri); };

// void DataTable::SetValue(KeyValue const& item) { Set(item.first, item.second, true); }
//
// void DataTable::SetValue(std::initializer_list<KeyValue> const& other) {
//    for (auto const& item : other) { Set(item.first, item.second, true); }
//}

int DataTable::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_data_) { f(item.first, item.second); }
    return Count();
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