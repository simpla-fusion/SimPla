//
// Created by salmon on 17-3-10.
//
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <map>
#include <memory>
#include <regex>
#include <string>
// Headers for SAMRAI
#include <SAMRAI/SAMRAI_config.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <simpla/data/DataUtility.h>
#include <simpla/data/all.h>
#include <simpla/toolbox/Log.h>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
// std::regex sub_group_regex;//(R"(([^/?#:]+)/)", std::regex::extended | std::regex::optimize);

struct DataBackendSAMRAI::pimpl_s {
    boost::shared_ptr<SAMRAI::tbox::Database> m_samrai_db_ = nullptr;
    static std::regex sub_group_regex;
    static std::regex match_path;

    typedef boost::shared_ptr<SAMRAI::tbox::Database> table_type;

    static std::shared_ptr<DataBackendSAMRAI> CreateBackend(boost::shared_ptr<SAMRAI::tbox::Database> const& db) {
        auto res = std::make_shared<DataBackendSAMRAI>();
        res->m_pimpl_->m_samrai_db_ = db;
        return res;
    };

    static std::shared_ptr<DataEntity> get_data_from_samrai(boost::shared_ptr<SAMRAI::tbox::Database> const& lobj);
    static void add_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj, std::string const& uri,
                                   std::shared_ptr<data::DataEntity> const& v);
    static void set_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj, std::string const& uri,
                                   std::shared_ptr<data::DataEntity> const& v);

    static std::pair<table_type, std::string> get_table(table_type self, std::string const& uri,
                                                        bool return_if_not_exist = true);
};

std::pair<DataBackendSAMRAI::pimpl_s::table_type, std::string> DataBackendSAMRAI::pimpl_s::get_table(
    table_type t, std::string const& uri, bool return_if_not_exist) {
    return HierarchicalTableForeach(t, uri, [&](table_type s_t, std::string const& k) { return s_t->isDatabase(k); },
                                    [&](table_type s_t, std::string const& k) { return s_t->getDatabase(k); },
                                    [&](table_type s_t, std::string const& k) {
                                        return return_if_not_exist ? static_cast<table_type>(nullptr)
                                                                   : s_t->putDatabase(k);
                                    });
};

DataBackendSAMRAI::DataBackendSAMRAI() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_samrai_db_ = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("");
}
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI const& other) : m_pimpl_(new pimpl_s) {
    UNSUPPORTED;
    //    m_pimpl_->m_samrai_db_ = boost::make_shared<SAMRAI::tbox::MemoryDatabase>(
    //        *boost::dynamic_pointer_cast<SAMRAI::tbox::MemoryDatabase>(other.m_pimpl_->m_samrai_db_));
}
DataBackendSAMRAI::DataBackendSAMRAI(std::string const& uri, std::string const& status) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_samrai_db_ = boost::make_shared<SAMRAI::tbox::MemoryDatabase>(uri);
}
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI&& other) : m_pimpl_(std::move(m_pimpl_)) {}

DataBackendSAMRAI::~DataBackendSAMRAI() {
    if (m_pimpl_->m_samrai_db_ != nullptr) { m_pimpl_->m_samrai_db_->close(); }
}
std::ostream& DataBackendSAMRAI::Print(std::ostream& os, int indent) const {
    m_pimpl_->m_samrai_db_->printClassData(os);
    return os;
}

boost::shared_ptr<SAMRAI::tbox::Database> DataBackendSAMRAI::db() { return m_pimpl_->m_samrai_db_; }
std::shared_ptr<DataBackend> DataBackendSAMRAI::Duplicate() const { return std::make_shared<DataBackendSAMRAI>(*this); }
std::shared_ptr<DataBackend> DataBackendSAMRAI::CreateNew() const { return std::make_shared<DataBackendSAMRAI>(); }

void DataBackendSAMRAI::Flush() { UNSUPPORTED; }
bool DataBackendSAMRAI::isNull() const { return m_pimpl_->m_samrai_db_ == nullptr; }
size_type DataBackendSAMRAI::size() const { return m_pimpl_->m_samrai_db_->getAllKeys().size(); }

// namespace detail {
void DataBackendSAMRAI::pimpl_s::set_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& dest,
                                                    std::string const& uri,
                                                    std::shared_ptr<data::DataEntity> const& src) {
    if (src->isTable()) {
        auto sub_db = uri == "" ? dest : dest->putDatabase(uri);
        src->cast_as<DataTable>().Accept([&](std::string const& k, std::shared_ptr<data::DataEntity> const& v) {
            set_data_to_samrai(sub_db, k, v);
        });
    } else if (uri == "") {
        return;
    } else if (src->isNull()) {
        dest->putDatabase(uri);
    } else if (src->isHeavyBlock()) {
    } else if (src->isArray()) {
        if (src->type() == typeid(bool)) {
            auto& varray = src->cast_as<DataArrayWrapper<bool>>().data();
            bool d[varray.size()];
            size_type num = varray.size();
            for (int i = 0; i < num; ++i) { d[i] = varray[i]; }
            dest->putBoolArray(uri, d, num);
        } else if (src->type() == typeid(std::string)) {
            auto& varray = src->cast_as<DataArrayWrapper<std::string>>().data();
            dest->putStringArray(uri, &varray[0], varray.size());
        } else if (src->type() == typeid(double)) {
            auto& varray = src->cast_as<DataArrayWrapper<double>>().data();
            dest->putDoubleArray(uri, &varray[0], varray.size());
        } else if (src->type() == typeid(int)) {
            auto& varray = src->cast_as<DataArrayWrapper<int>>().data();
            dest->putIntegerArray(uri, &varray[0], varray.size());
        } else if (src->cast_as<DataArray>().Get(0)->isArray() && src->cast_as<DataArray>().Get(0)->size() >= 3 &&
                   src->cast_as<DataArray>().Get(0)->type() == typeid(int)) {
            nTuple<int, 3> i_lo = data_cast<nTuple<int, 3>>(*src->cast_as<DataArray>().Get(0));
            nTuple<int, 3> i_up = data_cast<nTuple<int, 3>>(*src->cast_as<DataArray>().Get(1));

            SAMRAI::tbox::Dimension dim(3);
            dest->putDatabaseBox(uri, SAMRAI::tbox::DatabaseBox(dim, &(i_lo[0]), &(i_up[0])));
        }
    } else if (src->isLight()) {
        if (src->type() == typeid(bool)) {
            dest->putBool(uri, data_cast<bool>(*src));
        } else if (src->type() == typeid(std::string)) {
            dest->putString(uri, data_cast<std::string>(*src));
        } else if (src->type() == typeid(double)) {
            dest->putDouble(uri, data_cast<double>(*src));
        } else if (src->type() == typeid(int)) {
            dest->putInteger(uri, data_cast<int>(*src));
        }
    } else {
        WARNING << " Unknown type " << *src << " " << std::endl;
    }
}
void DataBackendSAMRAI::pimpl_s::add_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj,
                                                    std::string const& uri,
                                                    std::shared_ptr<data::DataEntity> const& v) {
    UNSUPPORTED;
}
std::shared_ptr<DataEntity> DataBackendSAMRAI::pimpl_s::get_data_from_samrai(
    boost::shared_ptr<SAMRAI::tbox::Database> const& lobj) {
    return std::make_shared<DataEntity>();
}

std::shared_ptr<DataEntity> DataBackendSAMRAI::Get(std::string const& uri) const {
    auto res = pimpl_s::get_table(m_pimpl_->m_samrai_db_, uri, true);
    return (res.first == nullptr || res.second == "")
               ? std::make_shared<DataEntity>()
               : pimpl_s::get_data_from_samrai(res.first->getDatabase(res.second));
}

void DataBackendSAMRAI::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    auto res = m_pimpl_->get_table(m_pimpl_->m_samrai_db_, uri, false);
    if (res.first != nullptr && res.second != "") { pimpl_s::set_data_to_samrai(res.first, res.second, v); }
}

void DataBackendSAMRAI::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    auto res = pimpl_s::get_table(m_pimpl_->m_samrai_db_, uri, false);
    if (res.second != "") { pimpl_s::add_data_to_samrai(res.first, res.second, v); }
}

size_type DataBackendSAMRAI::Delete(std::string const& uri) {
    auto res = pimpl_s::get_table(m_pimpl_->m_samrai_db_, uri, true);
    res.first->putDatabase(res.second);
    return 0;
}

size_type DataBackendSAMRAI::ForEach(
        std::function<void(std::string const &, std::shared_ptr<DataEntity>)> const &fun) const {
    auto keys = m_pimpl_->m_samrai_db_->getAllKeys();
    for (auto const& k : keys) { fun(k, pimpl_s::get_data_from_samrai(m_pimpl_->m_samrai_db_->getDatabase(k))); }
}

}  // namespace data{
}  // namespace simpla{