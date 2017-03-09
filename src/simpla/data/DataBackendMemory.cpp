//
// Created by salmon on 17-3-6.
//
#include "DataBackendMemory.h"
#include <iomanip>
#include <map>
#include "DataArray.h"
#include "DataEntity.h"
namespace simpla {
namespace data {
struct DataBackendMemory::pimpl_s {
    std::map<id_type, std::pair<std::string, std::shared_ptr<DataEntity>>> m_table_;
    static constexpr char split_char = '.';
    std::pair<std::shared_ptr<DataEntity>, std::string> insert_or_assign(std::string const& uri,
                                                                         std::shared_ptr<DataEntity>);
    id_type Hash(std::string const& s) const { return std::hash<std::string>()(s); }
};

DataBackendMemory::DataBackendMemory(std::string const& url, std::string const& status) : m_pimpl_(new pimpl_s) {
    if (url != "") { /* Open File */
    }
}
DataBackendMemory::DataBackendMemory(const DataBackendMemory& other) : m_pimpl_(new pimpl_s) {
    std::map<id_type, std::pair<std::string, std::shared_ptr<DataEntity>>>(other.m_pimpl_->m_table_)
        .swap(m_pimpl_->m_table_);
};
DataBackendMemory::DataBackendMemory(DataBackendMemory&& other) : m_pimpl_(new pimpl_s) {
    std::map<id_type, std::pair<std::string, std::shared_ptr<DataEntity>>>(other.m_pimpl_->m_table_)
        .swap(m_pimpl_->m_table_);
};
DataBackendMemory::~DataBackendMemory() {}

// std::ostream& print_kv(std::ostream& os, int indent, std::string const& key, std::shared_ptr<DataEntity> const& v) {
//    os << std::endl
//       << std::setw(indent + 1) << " "
//       << "\"" << key << "\" : ";
//    v->Print(os, indent + 1);
//    return os;
//}
//
// std::ostream& DataBackendMemory::Print(std::ostream& os, int indent) const {
//    if (!m_pimpl_->m_table_.empty()) {
//        auto it = m_pimpl_->m_table_.begin();
//        auto ie = m_pimpl_->m_table_.end();
//        if (it != ie) {
//            os << " {";
//            print_kv(os, indent + 1, it->first, it->second);
//            ++it;
//            for (; it != ie; ++it) {
//                os << " , ";
//                print_kv(os, indent + 1, it->first, it->second);
//            }
//            os << std::endl
//               << std::setw(indent) << " "
//               << " }";
//        }
//    };
//    return os;
//};
//
// std::pair<std::shared_ptr<DataEntity>, std::string> DataBackendMemory::pimpl_s::ParseURI(std::string const& str,
//                                                                                         bool create_table_if_need) {
//    size_type start_pos = 0;
//    size_type end_pos = str.size();
//    while (start_pos < end_pos) {
//        size_type pos0 = str.find(';', start_pos);
//        if (pos0 == std::string::npos) { pos0 = end_pos; }
//        std::string key = str.substr(start_pos, pos0 - start_pos);
//        size_type pos1 = key.find('=');
//        std::string value = "";
//        if (pos1 != std::string::npos) {
//            value = key.substr(pos1 + 1);
//            key = key.substr(0, pos1);
//        }
//        start_pos = pos0 + 1;
//    }
//}
std::unique_ptr<DataBackend> DataBackendMemory::CreateNew() const { return std::make_unique<DataBackendMemory>(); }
bool DataBackendMemory::IsNull() const { return m_pimpl_ == nullptr; };

std::shared_ptr<DataEntity> DataBackendMemory::Get(std::string const& url) const {
    std::shared_ptr<DataEntity> p = nullptr;
    size_type start_pos = 0, end_pos = 0;
    while (1) {
        end_pos = url.find(m_pimpl_->split_char, start_pos);
        std::string key =
            url.substr(start_pos, (end_pos == std::string::npos) ? std::string::npos : end_pos - start_pos);
        auto res = m_pimpl_->m_table_.find(m_pimpl_->Hash(key));
        if (res == m_pimpl_->m_table_.end()) {
            break;
        } else if (end_pos == std::string::npos) {
            p = res->second.second;
            break;
        } else if (!res->second.second->isTable()) {
            break;
        } else {
            start_pos = end_pos + 1;
        }
    }
    return p;
};

std::shared_ptr<DataEntity> DataBackendMemory::Get(id_type key) const {
    auto it = m_pimpl_->m_table_.find(key);
    return (it != m_pimpl_->m_table_.end()) ? it->second.second : std::make_shared<DataEntity>();
}

bool DataBackendMemory::Set(std::string const& url, std::shared_ptr<DataEntity> const& v) {
    bool success = false;
    size_type end_pos = url.find(m_pimpl_->split_char);
    if (end_pos == std::string::npos) {
        // insert or assign
        auto k = m_pimpl_->Hash(url);
        Set(k, v);
        m_pimpl_->m_table_.find(k)->second.first = url;
        success = true;
    } else {
        // find or create sub-table
        std::shared_ptr<DataTable> t = nullptr;
        std::string sub_k = url.substr(0, end_pos);
        id_type sub_id = m_pimpl_->Hash(sub_k);
        auto it = m_pimpl_->m_table_.find(sub_id);
        if (it == m_pimpl_->m_table_.end()) {
            // create table if need
            t = std::make_shared<DataTable>(this->CreateNew());
            m_pimpl_->m_table_.insert(std::make_pair(sub_id, std::make_pair(sub_k, t)));
        } else if (it->second.second->isTable()) {
            t = std::dynamic_pointer_cast<DataTable>(it->second.second);
        }
        if (t != nullptr) { success = t->Set(url.substr(end_pos + 1), v); }
    }

    return success;
};

bool DataBackendMemory::Set(id_type k, std::shared_ptr<DataEntity> const& v) {
    auto res = m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", v)));
    if (!res.second) { res.first->second.second = v; }
    return true;
}

bool DataBackendMemory::Add(std::string const& url, std::shared_ptr<DataEntity> const& v) {
    bool success = false;
    size_type end_pos = url.find(m_pimpl_->split_char);

    // insert or assign
    if (end_pos == std::string::npos) {
        id_type k = m_pimpl_->Hash(url);
        success = Add(k, v);
        m_pimpl_->m_table_.find(k)->second.first = url;
    } else {
        // find or create sub-table
        std::shared_ptr<DataTable> t = nullptr;
        std::string sub_k = url.substr(0, end_pos);
        id_type sub_id = m_pimpl_->Hash(sub_k);
        auto it = m_pimpl_->m_table_.find(sub_id);
        if (it == m_pimpl_->m_table_.end()) {
            // create table if need
            t = std::make_shared<DataTable>(this->CreateNew());
            m_pimpl_->m_table_.insert(std::make_pair(sub_id, std::make_pair(sub_k, t)));
        } else if (it->second.second->isTable()) {
            t = std::dynamic_pointer_cast<DataTable>(it->second.second);
        } else {
            success = false;
        }
        if (t != nullptr) { success = t->Add(url.substr(end_pos + 1), v); }
    }

    return success;
}
bool DataBackendMemory::Add(id_type k, std::shared_ptr<DataEntity> const& v) {
    //    auto res = m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", v)));
    //    if (!res.second) {
    //        auto p = res.first->second.second;
    //        if (p->isArray()) {
    //            std::dynamic_pointer_cast<DataArray>(res.first->second.second)->Add(v);
    //        } else {
    //            auto p = res.first->second.second->MakeArray();
    //            p->Add(v);
    //            res.first->second.second = p;
    //        }
    //    }

    auto it = m_pimpl_->m_table_.find(k);
    if (it == m_pimpl_->m_table_.end()) {
        if (v->isArray()) {
            m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", v)));
        } else {
            auto t_array = std::make_shared<DataArrayWrapper<void>>();
            t_array->Add(v);
            m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", t_array)));
        }
    } else {
        if (!it->second.second->isArray()) {
            auto p = std::make_shared<DataArrayWrapper<void>>();
            p->Add(it->second.second);
            it->second.second = std::dynamic_pointer_cast<DataEntity>(p);
        };
        it->second.second->cast_as<DataArray>().Add(v);
    }
    return true;
}

size_type DataBackendMemory::Delete(std::string const& uri) { return m_pimpl_->m_table_.erase(m_pimpl_->Hash(uri)); }
size_type DataBackendMemory::Delete(id_type key) { return m_pimpl_->m_table_.erase(key); }
void DataBackendMemory::DeleteAll() {}

size_t DataBackendMemory::Count(std::string const& uri) const {
    return uri == "" ? m_pimpl_->m_table_.size() : m_pimpl_->m_table_.count(m_pimpl_->Hash(uri));
}
size_type DataBackendMemory::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_table_) { f(item.second.first, item.second.second); }
}
size_type DataBackendMemory::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const {};

}  // namespace data {
}  // namespace simpla{