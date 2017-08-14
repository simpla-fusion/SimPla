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
#include "simpla/data/backend/DataUtility.h"

namespace simpla {
namespace data {
REGISTER_CREATOR(DataBackendMemory, mem);

struct DataBackendMemory::pimpl_s {
    typedef std::map<std::string, std::shared_ptr<DataEntity>> table_type;
    table_type m_table_;
    static std::pair<DataBackendMemory*, std::string> get_table(DataBackendMemory* self, std::string const& uri,
                                                                bool return_if_not_exist);
};

std::pair<DataBackendMemory*, std::string> DataBackendMemory::pimpl_s::get_table(DataBackendMemory* t,
                                                                                 std::string const& uri,
                                                                                 bool return_if_not_exist) {
    return HierarchicalTableForeach(
        t, uri,
        [&](DataBackendMemory* s_t, std::string const& k) -> bool {
            auto res = s_t->m_pimpl_->m_table_.find(k);
            return (res != s_t->m_pimpl_->m_table_.end()) &&
                   (dynamic_cast<data::DataTable const*>(res->second.get()) != nullptr);
        },
        [&](DataBackendMemory* s_t, std::string const& k) {
            return (std::dynamic_pointer_cast<DataBackendMemory>(
                        std::dynamic_pointer_cast<DataTable>(s_t->m_pimpl_->m_table_.find(k)->second)->backend())
                        .get());
        },
        [&](DataBackendMemory* s_t, std::string const& k) -> DataBackendMemory* {
            if (return_if_not_exist) { return nullptr; }
            auto res =
                s_t->m_pimpl_->m_table_.emplace(k, std::make_shared<DataTable>(std::make_shared<DataBackendMemory>()));
            return std::dynamic_pointer_cast<DataBackendMemory>(
                       std::dynamic_pointer_cast<DataTable>(res.first->second)->backend())
                .get();

        });
};

DataBackendMemory::DataBackendMemory() : m_pimpl_(new pimpl_s) {}
DataBackendMemory::DataBackendMemory(std::string const& url, std::string const& status) : DataBackendMemory() {
    if (url.empty()) {
        DataTable d(url);
        d.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> v) { Set(k, v); });
    }
}
DataBackendMemory::DataBackendMemory(const DataBackendMemory& other) : m_pimpl_(new pimpl_s) {
    std::map<std::string, std::shared_ptr<DataEntity>>(other.m_pimpl_->m_table_).swap(m_pimpl_->m_table_);
};
DataBackendMemory::DataBackendMemory(DataBackendMemory&& other) noexcept : m_pimpl_(new pimpl_s) {
    std::map<std::string, std::shared_ptr<DataEntity>>(other.m_pimpl_->m_table_).swap(m_pimpl_->m_table_);
};
DataBackendMemory::~DataBackendMemory() = default;

std::shared_ptr<DataBackend> DataBackendMemory::CreateNew() const { return std::make_shared<DataBackendMemory>(); }

std::shared_ptr<DataBackend> DataBackendMemory::Duplicate() const { return std::make_shared<DataBackendMemory>(*this); }

void DataBackendMemory::Flush() {}

std::ostream& DataBackendMemory::Print(std::ostream& os, int indent) const { return os; };

bool DataBackendMemory::isNull() const { return m_pimpl_ == nullptr; };
size_type DataBackendMemory::size() const { return m_pimpl_->m_table_.size(); }

std::shared_ptr<DataEntity> DataBackendMemory::Get(std::string const& url) const {
    std::shared_ptr<DataEntity> res = nullptr;
    auto t = m_pimpl_->get_table(const_cast<DataBackendMemory*>(this), url, false);
    if (t.first != nullptr && !t.second.empty()) {
        auto it = t.first->m_pimpl_->m_table_.find(t.second);
        if (it != t.first->m_pimpl_->m_table_.end()) { res = it->second; }
    }

    return res;
};

void DataBackendMemory::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    auto tab_res = pimpl_s::get_table((this), uri, true);
    if (tab_res.second.empty() || tab_res.first == nullptr) { return; }
    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, nullptr);
    if (res.first->second == nullptr) { res.first->second = v; }

    //    if (v->isTable()) {
    //        if (!overwrite && res.first->second != nullptr && !res.first->second->isTable()) {
    //            return 0;
    //        } else if (res.first->second == nullptr || !res.first->second->isTable()) {
    //            res.first->second = std::make_shared<DataTable>(std::make_shared<DataBackendMemory>());
    //        }
    //        auto& dest_table = res.first->second->cast_as<DataTable>();
    //        auto const& src_table = v->cast_as<DataTable>();
    //        src_table.Foreach(
    //            [&](std::string const& k, std::shared_ptr<DataEntity> const& v) { dest_table.Deserialize(k, v,
    //            overwrite);
    //            });
    //    } else if (v->isArray() && v->cast_as<DataArray>().isA(typeid(DataArrayWrapper<void>))) {
    //        auto dest_array = std::make_shared<DataArrayWrapper<void>>();
    //        auto const& src_array = v->cast_as<DataArray>();
    //        for (size_type i = 0, ie = src_array.size(); i < ie; ++i) { dest_array->Add(src_array.Serialize(i)); }
    //        res.first->second = dest_array;
    //    } else if (res.second || overwrite) {
    //        res.first->second = v;
    //    }
}
void DataBackendMemory::Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    auto tab_res = pimpl_s::get_table(this, uri, false);
    if (tab_res.second.empty()) { return; }
    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, std::make_shared<DataEntityWrapper<void*>>());
    if (dynamic_cast<DataArray const*>(res.first->second.get()) != nullptr &&
        res.first->second->value_type_info() == v->value_type_info()) {
    } else if (std::dynamic_pointer_cast<DataArrayWrapper<>>(res.first->second) != nullptr) {
        auto t_array = std::make_shared<DataArrayWrapper<>>();
        t_array->Add(res.first->second);
        res.first->second = t_array;
    }
    std::dynamic_pointer_cast<DataArray>(res.first->second)->Add(v);
}
size_type DataBackendMemory::Delete(std::string const& uri) {
    auto res = m_pimpl_->get_table(this, uri, true);
    return (res.first->m_pimpl_->m_table_.erase(res.second));
}

size_type DataBackendMemory::Foreach(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_table_) { f(item.first, item.second); }
    return 0;
}

}  // namespace data {
}  // namespace simpla{