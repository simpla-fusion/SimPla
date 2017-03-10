//
// Created by salmon on 17-3-6.
//
#include "DataBackendMemory.h"
#include <iomanip>
#include <map>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {

constexpr char DataBackendMemory::scheme_tag[];
std::string DataBackendMemory::scheme() const { return scheme_tag; }

struct DataBackendMemory::pimpl_s {
    std::map<id_type, std::pair<std::string, std::shared_ptr<DataEntity>>> m_table_;
    static constexpr char split_char = '.';
    id_type Hash(std::string const& s) const { return m_hasher_(s); }
    std::hash<std::string> m_hasher_;
};
DataBackendMemory::DataBackendMemory() : m_pimpl_(new pimpl_s) {}
DataBackendMemory::DataBackendMemory(std::string const& url, std::string const& status) : DataBackendMemory() {
    if (url != "") {
        DataTable d(url);
        d.Accept([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
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

DataBackend* DataBackendMemory::Create() const { return new DataBackendMemory; }

DataBackend* DataBackendMemory::Clone() const { return new DataBackendMemory(*this); }

void DataBackendMemory::Flush() {}

std::ostream& DataBackendMemory::Print(std::ostream& os, int indent) const { return os; };

bool DataBackendMemory::isNull() const { return m_pimpl_ == nullptr; };
size_type DataBackendMemory::size() const { return m_pimpl_->m_table_.size(); }

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
            t = std::make_shared<DataTable>(this->Clone());
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
            t = std::make_shared<DataTable>(this->Clone());
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

size_type DataBackendMemory::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_table_) { f(item.second.first, item.second.second); }
}
size_type DataBackendMemory::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const {};

}  // namespace data {
}  // namespace simpla{