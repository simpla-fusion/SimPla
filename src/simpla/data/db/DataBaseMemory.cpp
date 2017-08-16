//
// Created by salmon on 17-3-6.
//
#include "DataBaseMemory.h"
#include <iomanip>
#include <map>
#include <regex>
#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
#include "DataUtility.h"

namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseMemory, mem);

struct DataBaseMemory::pimpl_s {
    typedef std::map<std::string, std::shared_ptr<DataEntity>> table_type;
    table_type m_table_;
    static std::pair<DataBaseMemory*, std::string> get_table(DataBaseMemory* self, std::string const& uri,
                                                             bool return_if_not_exist);
};

std::pair<DataBaseMemory*, std::string> DataBaseMemory::pimpl_s::get_table(DataBaseMemory* t, std::string const& uri,
                                                                           bool return_if_not_exist) {
    return HierarchicalTableForeach(
        t, uri,
        [&](DataBaseMemory* s_t, std::string const& k) -> bool {
            auto res = s_t->m_pimpl_->m_table_.find(k);
            return (res != s_t->m_pimpl_->m_table_.end()) &&
                   (dynamic_cast<data::DataTable const*>(res->second.get()) != nullptr);
        },
        [&](DataBaseMemory* s_t, std::string const& k) {
            return (std::dynamic_pointer_cast<DataBaseMemory>(
                        std::dynamic_pointer_cast<DataTable>(s_t->m_pimpl_->m_table_.find(k)->second)->database())
                        .get());
        },
        [&](DataBaseMemory* s_t, std::string const& k) -> DataBaseMemory* {
            if (return_if_not_exist) { return nullptr; }
            auto res = s_t->m_pimpl_->m_table_.emplace(k, DataTable::New());
            return std::dynamic_pointer_cast<DataBaseMemory>(
                       std::dynamic_pointer_cast<DataTable>(res.first->second)->database())
                .get();

        });
};

DataBaseMemory::DataBaseMemory() : m_pimpl_(new pimpl_s) {}
DataBaseMemory::~DataBaseMemory() { delete m_pimpl_; };

int DataBaseMemory::Connect(std::string const& authority, std::string const& path, std::string const& query,
                            std::string const& fragment) {
    return SP_SUCCESS;
};
int DataBaseMemory::Disconnect() { return SP_SUCCESS; };
int DataBaseMemory::Flush() { return SP_SUCCESS; };

// std::ostream& DataBaseMemory::Print(std::ostream& os, int indent) const { return os; };

bool DataBaseMemory::isNull(std::string const& uri) const { return m_pimpl_ == nullptr; };
size_type DataBaseMemory::Count(std::string const& uri) const {
    return uri.empty() ? m_pimpl_->m_table_.size() : Get(uri)->Count();
}

std::shared_ptr<DataEntity> DataBaseMemory::Get(std::string const& url) const {
    std::shared_ptr<DataEntity> res = nullptr;
    auto t = m_pimpl_->get_table(const_cast<DataBaseMemory*>(this), url, false);
    if (t.first != nullptr && !t.second.empty()) {
        auto it = t.first->m_pimpl_->m_table_.find(t.second);
        if (it != t.first->m_pimpl_->m_table_.end()) { res = it->second; }
    }

    return res;
};

int DataBaseMemory::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    auto tab_res = pimpl_s::get_table((this), uri, true);
    if (tab_res.second.empty() || tab_res.first == nullptr) { return 0; }
    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, nullptr);
    if (res.first->second == nullptr) { res.first->second = v; }

    //    if (v->isTable()) {
    //        if (!overwrite && res.first->second != nullptr && !res.first->second->isTable()) {
    //            return 0;
    //        } else if (res.first->second == nullptr || !res.first->second->isTable()) {
    //            res.first->second = std::make_shared<DataTable>(std::make_shared<DataBaseMemory>());
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
    return 1;
}
int DataBaseMemory::Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    auto tab_res = pimpl_s::get_table(this, uri, false);
    if (tab_res.second.empty()) { return 0; }
    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, DataArray::New());
    if (dynamic_cast<DataArray const*>(res.first->second.get()) != nullptr &&
        res.first->second->value_type_info() == v->value_type_info()) {
    } else if (std::dynamic_pointer_cast<DataArray>(res.first->second) != nullptr) {
        auto t_array = DataArray::New();
        t_array->Add(res.first->second);
        res.first->second = t_array;
    }
    std::dynamic_pointer_cast<DataArray>(res.first->second)->Add(v);
    return 1;
}
int DataBaseMemory::Delete(std::string const& uri) {
    auto res = m_pimpl_->get_table(this, uri, true);
    return (res.first->m_pimpl_->m_table_.erase(res.second));
}

int DataBaseMemory::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    int counter = 0;
    for (auto const& item : m_pimpl_->m_table_) { counter += f(item.first, item.second); }
    return counter;
}

}  // namespace data {
}  // namespace simpla{