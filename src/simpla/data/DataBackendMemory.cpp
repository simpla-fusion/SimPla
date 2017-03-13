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
#include "DataUtility.h"

namespace simpla {
namespace data {

constexpr char DataBackendMemory::scheme_tag[];
std::string DataBackendMemory::scheme() const { return scheme_tag; }

struct DataBackendMemory::pimpl_s {
    typedef std::map<std::string, std::shared_ptr<DataEntity>> table_type;
    table_type m_table_;
    static std::pair<table_type*, std::string> get_table(table_type* self, std::string const& uri,
                                                         bool return_if_not_exist = true);
};

std::pair<DataBackendMemory::pimpl_s::table_type*, std::string> DataBackendMemory::pimpl_s::get_table(
    table_type* t, std::string const& uri, bool return_if_not_exist) {
    return HierarchicalGetTable(t, uri,
                                [&](table_type *s_t, std::string const &k)
                                {
                                    auto res = s_t->find(k);
                                    return res != s_t->end() && res->second->isTable();
                                },
                                [&](table_type *s_t, std::string const &k)
                                {
                                    return &(std::dynamic_pointer_cast<DataTable>(s_t->find(k)->second)
                                            ->backend()
                                            ->cast_as<DataBackendMemory>()
                                            .m_pimpl_->m_table_);
                                },
                                [&](table_type *s_t, std::string const &k)
                                {
                                    if (return_if_not_exist) { return static_cast<table_type *>(nullptr); }
                                    return &(s_t->emplace(k, std::make_shared<DataTable>(
                                                    std::make_shared<DataBackendMemory>()))
                                            .first->second->cast_as<DataTable>()
                                            .backend()
                                            ->cast_as<DataBackendMemory>()
                                            .m_pimpl_->m_table_);

                                });
};

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

std::shared_ptr<DataBackend> DataBackendMemory::CreateNew() const { return std::make_shared<DataBackendMemory>(); }

std::shared_ptr<DataBackend> DataBackendMemory::Duplicate() const { return std::make_shared<DataBackendMemory>(*this); }

void DataBackendMemory::Flush() {}

std::ostream& DataBackendMemory::Print(std::ostream& os, int indent) const { return os; };

bool DataBackendMemory::isNull() const { return m_pimpl_ == nullptr; };
size_type DataBackendMemory::size() const { return m_pimpl_->m_table_.size(); }

std::shared_ptr<DataEntity> DataBackendMemory::Get(std::string const& url) const {
    auto res = m_pimpl_->get_table(&(m_pimpl_->m_table_), url);
    return res.first != nullptr ? res.first->at(res.second) : std::make_shared<DataEntity>();
};

void DataBackendMemory::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    auto tab_res = pimpl_s::get_table(&(m_pimpl_->m_table_), uri, false);
    if (tab_res.second != "") {
        auto res = tab_res.first->emplace(tab_res.second, v);
        if (!res.second) { res.first->second = v; }
    }
}
void DataBackendMemory::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    auto tab_res = pimpl_s::get_table(&(m_pimpl_->m_table_), uri, false);
    if (tab_res.second != "") {
        auto res = tab_res.first->emplace(tab_res.second, v);
        if (!res.second) {
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
size_type DataBackendMemory::Delete(std::string const& uri) {
    auto res = m_pimpl_->get_table(&(m_pimpl_->m_table_), uri);
    return (res.first != nullptr && res.second != "") ? res.first->erase(res.second) : 0;
}

size_type DataBackendMemory::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    for (auto const& item : m_pimpl_->m_table_) { f(item.first, item.second); }
}

}  // namespace data {
}  // namespace simpla{