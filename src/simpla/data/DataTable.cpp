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
struct DataTable::pimpl_s {
    std::map<std::string, std::shared_ptr<DataEntity> > m_table_;

    std::shared_ptr<DataEntity> emplace(DataTable* t, std::string const& url,
                                        std::shared_ptr<DataEntity> p = nullptr,
                                        char split_char = '.');

    DataEntity const* search(DataTable const*, std::string const& url, char split_char = '.');
};

std::shared_ptr<DataEntity> DataTable::pimpl_s::emplace(DataTable* t, std::string const& url,
                                                        std::shared_ptr<DataEntity> p,
                                                        char split_char) {
    size_type start_pos = 0;
    size_type end_pos = url.size();
    while (start_pos < end_pos) {
        size_type pos = url.find(split_char, start_pos);

        if (pos != std::string::npos) {
            auto res = t->m_pimpl_->m_table_.emplace(
                url.substr(start_pos, pos - start_pos),
                std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>()));

            if (!res.first->second->is_table()) {
                p = nullptr;
                break;
            } else if (pos == end_pos - 1) {
                p = res.first->second;
                break;
            }

            t = &res.first->second.get()->as_table();
            start_pos = pos + 1;
            continue;

        } else {
            auto res = t->m_pimpl_->m_table_.emplace(url.substr(start_pos), p);

            p = res.first->second;
            break;
        }
    }
    if (p == nullptr) RUNTIME_ERROR << " Can not insert entity at [" << url << "]" << std::endl;

    return p;
};

DataEntity const* DataTable::pimpl_s::search(DataTable const* t, std::string const& url,
                                             char split_char) {
    size_type start_pos = 0;
    size_type end_pos = url.size();
    while (start_pos < end_pos) {
        size_type pos = url.find(split_char, start_pos);

        if (pos != std::string::npos) {
            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos, pos - start_pos));

            if (pos == end_pos - 1) {
                return it->second.get();
            } else if (it == t->m_pimpl_->m_table_.end() || !it->second->is_table()) {
                break;
            }

            t = &it->second->as_table();
            start_pos = pos + 1;
            continue;

        } else {
            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos));

            if (it != t->m_pimpl_->m_table_.end()) { return it->second.get(); }

            break;
        }
    }

    return nullptr;
};

DataTable::DataTable() : DataEntity(), m_pimpl_(new pimpl_s){};

DataTable::DataTable(std::initializer_list<KeyValue> const& t) : DataTable() { insert(t); };

// DataTable::DataTable(DataTable const& other) : m_pimpl_(new pimpl_s(other.m_pimpl_)){};

DataTable::DataTable(DataTable&& other) : m_pimpl_(other.m_pimpl_){};

DataTable::~DataTable(){};

void DataTable::insert(std::initializer_list<KeyValue> const& others) {
    for (auto const& kv : others) { insert(kv); }
}
std::ostream& print_kv(std::ostream& os, int indent, std::string const& k, DataEntity const& v) {
    if (v.is_table()) { os << std::endl << std::setw(indent + 1) << " "; }
    os << k << " = " << v;
    return os;
}

std::ostream& DataTable::print(std::ostream& os, int indent) const {
    if (!DataEntity::is_null()) { DataEntity::print(os, indent + 1); }

    if (!m_pimpl_->m_table_.empty()) {
        auto it = m_pimpl_->m_table_.begin();
        auto ie = m_pimpl_->m_table_.end();
        if (it != ie) {
            os << "{ ";
            print_kv(os, indent, it->first, *it->second);
            //            os << it->first << " = " << *it->second;
            ++it;
            for (; it != ie; ++it) {
                os << " , ";
                print_kv(os, indent, it->first, *it->second);
                // os << " , " << it->first << " = " << *it->second;
            }

            os << " }";
        }
    }
    return os;
};

bool DataTable::empty() const { return (m_pimpl_ != nullptr) && m_pimpl_->m_table_.empty(); };

bool DataTable::has(std::string const& url) const { return find(url) != nullptr; };

DataTable* DataTable::create_table(std::string const& url) {
    return &(m_pimpl_->emplace(this, url + ".")->as_table());
}

std::shared_ptr<DataEntity> DataTable::set_value(std::string const& url,
                                                 std::shared_ptr<DataEntity> const& v) {
    return m_pimpl_->emplace(this, url, v);
};

std::shared_ptr<DataEntity> DataTable::get(std::string const& url) {
    return m_pimpl_->emplace(this, url);
}

void DataTable::parse(std::string const& str) {
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

        if (value == "") {
            set_value(key, true);
        } else {
            set_value(key, value);
        }

        start_pos = pos0 + 1;
    }
}

DataEntity const* DataTable::find(std::string const& url) const {
    return m_pimpl_->search(this, url);
};

DataEntity& DataTable::at(std::string const& url) {
    DataEntity* res = const_cast<DataEntity*>(find(url));

    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; }

    return *res;
};

DataEntity const& DataTable::at(std::string const& url) const {
    DataEntity const* res = find(url);
    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; }
    return *res;
};

void DataTable::foreach (
    std::function<void(std::string const&, DataEntity const&)> const& fun) const {
    for (auto& item : m_pimpl_->m_table_) {
        fun(item.first, *std::dynamic_pointer_cast<const DataEntity>(item.second));
    }
}

void DataTable::foreach (std::function<void(std::string const&, DataEntity&)> const& fun) {
    for (auto& item : m_pimpl_->m_table_) { fun(item.first, *item.second); }
};
}
}  // namespace simpla{namespace toolbox{