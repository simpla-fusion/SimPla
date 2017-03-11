//
// Created by salmon on 17-3-6.
//
#include "DataBackendMemory.h"
#include <iomanip>
#include <map>
#include <regex>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {

constexpr char DataBackendMemory::scheme_tag[];
std::string DataBackendMemory::scheme() const { return scheme_tag; }

struct DataBackendMemory::pimpl_s {
    std::map<std::string, std::shared_ptr<DataEntity>> m_table_;
    static constexpr char split_char = '.';
    static void add_or_set(DataBackendMemory* self, std::string const& uri, std::shared_ptr<DataEntity> const& v,
                           bool do_add);
    static std::regex uri_regex;  //(R"(^(/(([^/?#:]+)/)*)*([^/?#:]*)$)", std::regex::extended | std::regex::optimize);
};
std::regex DataBackendMemory::pimpl_s::uri_regex(R"(^(/(([^/?#:]+)/)*)*([^/?#:]*)$)",
                                                 std::regex::extended | std::regex::optimize);

DataBackendMemory::DataBackendMemory() : m_pimpl_(new pimpl_s) {}
DataBackendMemory::DataBackendMemory(std::string const& url, std::string const& status) : DataBackendMemory() {
    if (url != "") {
        DataTable d(url);
        d.Accept([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
    }
}
DataBackendMemory::DataBackendMemory(const DataBackendMemory& other) : m_pimpl_(new pimpl_s) {
    std::map<std::string, std::shared_ptr<DataEntity>>(other.m_pimpl_->m_table_).swap(m_pimpl_->m_table_);
};
DataBackendMemory::DataBackendMemory(DataBackendMemory&& other) : m_pimpl_(new pimpl_s) {
    std::map<std::string, std::shared_ptr<DataEntity>>(other.m_pimpl_->m_table_).swap(m_pimpl_->m_table_);
};
DataBackendMemory::~DataBackendMemory() {}

std::shared_ptr<DataBackend> DataBackendMemory::Create() const { return std::make_shared<DataBackendMemory>(); }

std::shared_ptr<DataBackend> DataBackendMemory::Clone() const { return std::make_shared<DataBackendMemory>(*this); }

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
        auto res = m_pimpl_->m_table_.find(key);
        if (res == m_pimpl_->m_table_.end()) {
            break;
        } else if (end_pos == std::string::npos) {
            p = res->second;
            break;
        } else if (!res->second->isTable()) {
            break;
        } else {
            start_pos = end_pos + 1;
        }
    }
    return p;
};

// std::shared_ptr<DataTable> DataBackendMemory::pimpl_s::add_table(DataBackendMemory* self, std::string const& uri) {
//    static std::regex sub_dir_regex(R"(([^/?#:]+)/)", std::regex::extended | std::regex::optimize);
//
//    std::smatch sub_dir_match_result;
//
//    auto pos = uri.begin();
//    auto end = uri.end();
//    auto* t = &(self->m_pimpl_->m_table_);
//
//    std::shared_ptr<DataTable> t_table = std::make_shared<DataTable>(self->Create());
//    std::shared_ptr<DataTable> result(nullptr);
//    for (; std::regex_search(pos, end, sub_dir_match_result, sub_dir_regex);
//         pos = sub_dir_match_result.suffix().first) {
//        auto k = sub_dir_match_result.str(1);
//        auto res = t->emplace(k, std::dynamic_pointer_cast<DataEntity>(t_table));
//        if (res.second) { t_table = std::make_shared<DataTable>(self->Create()); }
//        if (!res.first->second->isTable()) {
//            RUNTIME_ERROR << std::endl
//                          << std::setw(25) << std::right << "illegal path [/" << uri << "]" << std::endl
//                          << std::setw(25) << std::right << "     at here   " << std::setw(&(*pos) - &(*uri.begin()))
//                          << " "
//                          << " ^" << std::endl;
//        } else {
//            result = std::dynamic_pointer_cast<DataTable>(res.first->second);
//            t = &(result->backend()->cast_as<DataBackendMemory>().m_pimpl_->m_table_);
//        }
//    }
//    return result;
//}
void DataBackendMemory::pimpl_s::add_or_set(DataBackendMemory* self, std::string const& uri,
                                            std::shared_ptr<DataEntity> const& v, bool do_add) {
    std::smatch uri_match_result;

    if (!std::regex_match(uri, uri_match_result, DataBackendMemory::pimpl_s::uri_regex)) {
        RUNTIME_ERROR << " illegal uri : [" << uri << "]" << std::endl;
    }

    auto* t = &(self->m_pimpl_->m_table_);

    if (uri_match_result[1].length() != 0) {
        static std::regex sub_dir_regex(R"(([^/?#:]+)/)", std::regex::extended | std::regex::optimize);

        std::smatch sub_dir_match_result;

        auto pos = uri.begin();
        auto end = uri.end();

        std::shared_ptr<DataTable> t_table = std::make_shared<DataTable>(self->Create());
        std::shared_ptr<DataTable> result(nullptr);
        for (; std::regex_search(pos, end, sub_dir_match_result, sub_dir_regex);
             pos = sub_dir_match_result.suffix().first) {
            auto k = sub_dir_match_result.str(1);
            auto res = t->emplace(k, std::dynamic_pointer_cast<DataEntity>(t_table));
            if (res.second) { t_table = std::make_shared<DataTable>(self->Create()); }
            if (!res.first->second->isTable()) {
                RUNTIME_ERROR << std::endl
                              << std::setw(25) << std::right << "illegal path [/" << uri << "]" << std::endl
                              << std::setw(25) << std::right << "     at here   "
                              << std::setw(&(*pos) - &(*uri.begin())) << " "
                              << " ^" << std::endl;
            } else {
                result = std::dynamic_pointer_cast<DataTable>(res.first->second);
                t = &(result->backend()->cast_as<DataBackendMemory>().m_pimpl_->m_table_);
            }
        }
    }
    if (uri_match_result.str(4) != "") {
        auto res = t->emplace(uri_match_result.str(4), v);
        if (!res.second) {
            if (!do_add || res.first->second == nullptr) {
                res.first->second = v;
            } else {
                if (!res.first->second->isArray()) {
                    auto t_array = std::make_shared<DataArrayWrapper<void>>();
                    t_array->Add(res.first->second);
                    t_array->Add(v);
                    res.first->second = t_array;
                } else {
                    std::dynamic_pointer_cast<DataArray>(res.first->second)->Add(v);
                }
            }
        }
    }
}

void DataBackendMemory::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    m_pimpl_->add_or_set(this, uri, v, false);
}
void DataBackendMemory::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    m_pimpl_->add_or_set(this, uri, v, true);
}
//
// bool DataBackendMemory::Add(std::string const& url, std::shared_ptr<DataEntity> const& v) {
//    bool success = false;
//    size_type end_pos = url.find(m_pimpl_->split_char);
//
//    // insert or assign
//    if (end_pos == std::string::npos) {
//        success = Add(url, v);
//    } else {
//        // find or create sub-table
//        std::shared_ptr<DataTable> t = nullptr;
//        std::string sub_k = url.substr(0, end_pos);
//        auto it = m_pimpl_->m_table_.find(sub_k);
//        if (it == m_pimpl_->m_table_.end()) {
//            // create table if need
//            t = std::make_shared<DataTable>(this->Clone());
//            m_pimpl_->m_table_.insert(std::make_pair(sub_k, t));
//        } else if (it->second->isTable()) {
//            t = std::dynamic_pointer_cast<DataTable>(it->second);
//        } else {
//            success = false;
//        }
//        if (t != nullptr) { success = t->Add(url.substr(end_pos + 1), v); }
//    }
//
//    return success;
//}
//    bool success = false;
//
//    //    size_type end_pos = url.find(m_pimpl_->split_char);
//    //    if (end_pos == std::string::npos) {
//    //        // insert or assign
//    //        auto k = m_pimpl_->Hash(url);
//    //        Set(k, v);
//    //        m_pimpl_->m_table_.find(k)->second.first = url;
//    //        success = true;
//    //    } else {
//    //        // find or create sub-table
//    //        std::shared_ptr<DataTable> t = nullptr;
//    //        std::string sub_k = url.substr(0, end_pos);
//    //        id_type sub_id = m_pimpl_->Hash(sub_k);
//    //        auto it = m_pimpl_->m_table_.find(sub_id);
//    //        if (it == m_pimpl_->m_table_.end()) {
//    //            // create table if need
//    //            t = std::make_shared<DataTable>(this->Clone());
//    //            m_pimpl_->m_table_.insert(std::make_pair(sub_id, std::make_pair(sub_k, t)));
//    //        } else if (it->second.second->isTable()) {
//    //            t = std::dynamic_pointer_cast<DataTable>(it->second.second);
//    //        }
//    //        if (t != nullptr) { success = t->Set(url.substr(end_pos + 1), v); }
//    //    }
//
//    return success;
//};
// bool DataBackendMemory::Add(id_type k, std::shared_ptr<DataEntity> const& v, std::string const& name) {
//    //    auto res = m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", v)));
//    //    if (!res.second) {
//    //        auto p = res.first->second.second;
//    //        if (p->isArray()) {
//    //            std::dynamic_pointer_cast<DataArray>(res.first->second.second)->Add(v);
//    //        } else {
//    //            auto p = res.first->second.second->MakeArray();
//    //            p->Add(v);
//    //            res.first->second.second = p;
//    //        }
//    //    }
//
//    auto it = m_pimpl_->m_table_.find(k);
//    if (it == m_pimpl_->m_table_.end()) {
//        if (v->isArray()) {
//            m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", v)));
//        } else {
//            auto t_array = std::make_shared<DataArrayWrapper<void>>();
//            t_array->Add(v);
//            m_pimpl_->m_table_.insert(std::make_pair(k, std::make_pair("", t_array)));
//        }
//    } else {
//        if (!it->second.second->isArray()) {
//            auto p = std::make_shared<DataArrayWrapper<void>>();
//            p->Add(it->second.second);
//            it->second.second = std::dynamic_pointer_cast<DataEntity>(p);
//        };
//        it->second.second->cast_as<DataArray>().Add(v);
//    }
//    return true;
//}

size_type DataBackendMemory::Delete(std::string const& uri) { return m_pimpl_->m_table_.erase(uri); }

size_type DataBackendMemory::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_table_) { f(item.first, item.second); }
}

}  // namespace data {
}  // namespace simpla{