//
// Created by salmon on 16-11-9.
//
#include "DataTable.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>
#include <string>
#include "DataEntity.h"

namespace simpla {
namespace data {

struct KeyValue::pimpl_s {
    std::string m_key_;
    std::shared_ptr<DataEntity> m_value_;
};
KeyValue::KeyValue(unsigned long long int n, std::shared_ptr<DataEntity> const& p) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_key_ = std::to_string(n);
    m_pimpl_->m_value_ = p;
}

KeyValue::KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_key_ = k;
    m_pimpl_->m_value_ = p;
}

KeyValue::KeyValue(KeyValue const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_key_ = other.m_pimpl_->m_key_;
    m_pimpl_->m_value_ = other.m_pimpl_->m_value_;
}
KeyValue::KeyValue(KeyValue&& other) : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(nullptr); }
KeyValue::~KeyValue() {}

void KeyValue::SetValue(std::shared_ptr<DataEntity> const& p) { m_pimpl_->m_value_ = p; }

std::string const& KeyValue::key() const { return m_pimpl_->m_key_; }
DataEntity const& KeyValue::value() const { return *m_pimpl_->m_value_; }
DataEntity const* KeyValue::pointer() const { return m_pimpl_->m_value_.get(); }

struct DataTable::pimpl_s {
    static const char split_char = '.';
    std::map<id_type, KeyValue> m_table_;

    KeyValue* insert(DataTable* t, KeyValue const& p, std::string const& url = "");

    KeyValue* search(DataTable const*, std::string const& url);
};

KeyValue* DataTable::pimpl_s::insert(DataTable* t, KeyValue const& p0, std::string const& url) {
//    size_type start_pos = 0;
//    size_type end_pos = url.size();
//    while (start_pos < end_pos) {
//        size_type pos = url.find(split_char, start_pos);
//
//        if (pos != std::string::npos) {
//            auto res =
//                t->m_pimpl_->m_table_.emplace(url.substr(start_pos, pos - start_pos),
//                                              std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>()));
//            if (!res.first->second.value().isTable()) {
//                break;
//            } else if (pos == end_pos - 1) {
//                p = res.first->second;
//                break;
//            }
//            t = &(res.first->second.as<DataTable>());
//            start_pos = pos + 1;
//            continue;
//        } else if (p0.value().isTable()) {
//            auto res = t->m_pimpl_->m_table_.emplace(url.substr(start_pos), p);
//            p = res.first->second;
//            break;
//        } else {
////            t->m_pimpl_->m_table_[url.substr(start_pos)] = p;
//            break;
//        }
//    }
////    if (p.empty()) RUNTIME_ERROR << " Can not Connect entity at [" << url << "]" << std::endl;


};

KeyValue* DataTable::pimpl_s::search(DataTable const* t, std::string const& url) {
//    size_type start_pos = 0;
//    size_type end_pos = url.size();
//    while (start_pos < end_pos) {
//        size_type pos = url.find(split_char, start_pos);
//
//        if (pos != std::string::npos) {
//            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos, pos - start_pos));
//
//            if (pos == end_pos - 1) {
//                return it->second;
//            } else if (it == t->m_pimpl_->m_table_.end() || !it->second.isTable()) {
//                break;
//            }
//
//            t = &(it->second.as<DataTable>());
//            start_pos = pos + 1;
//            continue;
//
//        } else {
//            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos));
//
//            if (it != t->m_pimpl_->m_table_.end()) { return it->second; }
//
//            break;
//        }
//    }

    return nullptr;
};

DataTable::DataTable() : DataEntity(), m_pimpl_(new pimpl_s){};
DataTable::DataTable(DataTable&& other) : m_pimpl_(other.m_pimpl_){};
DataTable::DataTable(DataTable const& other) : m_pimpl_(other.m_pimpl_){};
DataTable::DataTable(std::initializer_list<KeyValue> const& others) : DataTable() { Set(others); }
DataTable::~DataTable(){};

std::ostream& print_kv(std::ostream& os, int indent, KeyValue const& v) {
    if (v.value().isTable()) { os << std::endl << std::setw(indent + 1) << " "; }
    //    os << k << " = " << v;
    return os;
}

std::ostream& DataTable::Print(std::ostream& os, int indent) const {
    //    if (!DataEntity::isNull()) { DataEntity::Print(os, indent + 1); }

    if (!m_pimpl_->m_table_.empty()) {
        auto it = m_pimpl_->m_table_.begin();
        auto ie = m_pimpl_->m_table_.end();
        if (it != ie) {
            os << "{ ";
            print_kv(os, indent, it->second);
            //            os << it->first << " = " << *it->second;
            ++it;
            for (; it != ie; ++it) {
                os << " , ";
                print_kv(os, indent, it->second);
                // os << " , " << it->first << " = " << *it->second;
            }

            os << " }";
        }
    };
    return os;
};

bool DataTable::empty() const { return (m_pimpl_ != nullptr) && m_pimpl_->m_table_.empty(); };

bool DataTable::has(std::string const& url) const { return find(url) != nullptr; };

KeyValue* DataTable::CreateTable(std::string const& url) {
    //    auto p = m_pimpl_->insert(this, url + ".");
    //    ASSERT(p.isTable());
    return nullptr;  // std::dynamic_pointer_cast<DataTable>(p);
}

// void DataTable::Set(std::string const& url, std::shared_ptr<DataEntity> const& v) {  };

KeyValue* DataTable::Get(std::string const& url) { return nullptr; /*m_pimpl_->insert(this, url);*/ }
KeyValue const* DataTable::Get(std::string const& url) const { return nullptr; /*m_pimpl_->search(this, url);*/ }

KeyValue* DataTable::Set(KeyValue const& k_v, std::string const& prefix) {
    return m_pimpl_->insert(this, k_v, prefix);
};
void DataTable::Set(std::initializer_list<KeyValue> const& c) {
    for (auto const& item : c) { Set(item); }
}
void DataTable::ParseFile(std::string const& str) { UNIMPLEMENTED; }
void DataTable::Parse(std::string const& str) {
    size_type start_pos = 0;
    size_type end_pos = str.size();
    while (start_pos < end_pos) {
        size_type pos0 = str.find(';', start_pos);
        if (pos0 == std::string::npos) { pos0 = end_pos; }
        std::string key = str.substr(start_pos, pos0 - start_pos);
        size_type pos1 = key.find('=');
        std::string value = "";
        if (pos1 != std::string::npos) {
            value = key.substr(pos1 + 1);
            key = key.substr(0, pos1);
        }

        //        if (value == "") {
        //            SetValue(key, true);
        //        } else {
        //            SetValue(key, value);
        //        }
        start_pos = pos0 + 1;
    }
}
KeyValue* DataTable::find(std::string const& url) const { return nullptr; /*return m_pimpl_->search(this, url);*/ };
// void DataTable::Merge(DataTable const& other) { UNIMPLEMENTED; }

// DataEntity& DataTable::at(std::string const& url) {
//    DataEntity* res = const_cast<DataEntity*>(find(url));
//
//    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; }
//
//    return *res;
//};
//
// DataEntity const& DataTable::at(std::string const& url) const {
//    DataEntity const* res = find(url);
//    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; }
//    return *res;
//};
//
// void DataTable::foreach (std::function<void(std::string const&, DataEntity const&)> const& fun) const {
//    for (auto& item : m_pimpl_->m_table_) {
//        fun(item.first, *std::dynamic_pointer_cast<const DataEntity>(item.second));
//    }
//}
//
// void DataTable::foreach (std::function<void(std::string const&, DataEntity&)> const& fun) {
//    for (auto& item : m_pimpl_->m_table_) { fun(item.first, *item.second); }
//};
//
// std::shared_ptr<DataEntity> CreateDataEntity(const std::initializer_list<simpla::data::KeyValue> const& kvs) {
//    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(kvs));
//};
}
}  // namespace simpla{namespace toolbox{