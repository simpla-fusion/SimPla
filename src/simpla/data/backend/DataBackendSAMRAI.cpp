//
// Created by salmon on 17-3-10.
//
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <map>
#include <memory>
#include <string>
// Headers for SAMRAI
#include <SAMRAI/SAMRAI_config.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
constexpr char DataBackendSAMRAI::ext[];

struct DataBackendSAMRAI::pimpl_s {
    boost::shared_ptr<SAMRAI::tbox::Database> m_samrai_db_ = nullptr;

    static std::shared_ptr<DataEntity> get_data_from_samrai(boost::shared_ptr<SAMRAI::tbox::Database> const& lobj);

    static void add_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj, std::string const& uri,
                                   std::shared_ptr<data::DataEntity> const& v);
    static void set_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj, std::string const& uri,
                                   std::shared_ptr<data::DataEntity> const& v);
};
DataBackendSAMRAI::DataBackendSAMRAI() : m_pimpl_(new pimpl_s) {}
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI const& other) : DataBackendSAMRAI() {} /* copy pimpl_s*/
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendSAMRAI::DataBackendSAMRAI(std::string const& uri, std::string const& status) : DataBackendSAMRAI() {
    m_pimpl_->m_samrai_db_ = boost::make_shared<SAMRAI::tbox::MemoryDatabase>(uri);
}
DataBackendSAMRAI::~DataBackendSAMRAI() {
    if (m_pimpl_->m_samrai_db_ != nullptr) { m_pimpl_->m_samrai_db_->close(); }
}
std::ostream& DataBackendSAMRAI::Print(std::ostream& os, int indent) const {
    m_pimpl_->m_samrai_db_->printClassData(os);
    return os;
}
std::unique_ptr<DataBackend> DataBackendSAMRAI::CreateNew() const { return std::make_unique<DataBackendSAMRAI>(); }
void DataBackendSAMRAI::Flush() { UNSUPPORTED; }
bool DataBackendSAMRAI::isNull() const { return m_pimpl_->m_samrai_db_ == nullptr; }
size_type DataBackendSAMRAI::size() const { return m_pimpl_->m_samrai_db_->getAllKeys().size(); }
//
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
    } else if (src->type() == typeid(bool)) {
        if (src->isArray()) {
            auto& varray = src->cast_as<DataArrayWrapper<bool>>().data();
            bool d[varray.size()];
            size_type num = varray.size();
            for (int i = 0; i < num; ++i) { d[i] = varray[i]; }
            dest->putBoolArray(uri, d, num);
        } else {
            dest->putBool(uri, src->as<bool>());
        }
    } else if (src->type() == typeid(std::string)) {
        if (src->isArray()) {
            auto& varray = src->cast_as<DataArrayWrapper<std::string>>().data();
            dest->putStringArray(uri, &varray[0], varray.size());
        } else {
            dest->putString(uri, src->as<std::string>());
        }
    } else if (src->type() == typeid(double)) {
        if (src->isArray()) {
            auto& varray = src->cast_as<DataArrayWrapper<double>>().data();
            dest->putDoubleArray(uri, &varray[0], varray.size());
        } else {
            dest->putDouble(uri, src->as<double>());
        }
    } else if (src->type() == typeid(int)) {
        if (src->isArray()) {
            auto& varray = src->cast_as<DataArrayWrapper<int>>().data();
            dest->putIntegerArray(uri, &varray[0], varray.size());
        } else {
            dest->putInteger(uri, src->as<int>());
        }
    } else if (src->type() == typeid(nTuple<bool, 3>)) {
        dest->putBoolArray(uri, &src->as<nTuple<bool, 3>>()[0], 3);
    } else if (src->type() == typeid(nTuple<int, 3>)) {
        dest->putIntegerArray(uri, &src->as<nTuple<int, 3>>()[0], 3);
    } else if (src->type() == typeid(nTuple<double, 3>)) {
        dest->putDoubleArray(uri, &src->as<nTuple<double, 3>>()[0], 3);
    } else if (src->type() == typeid(std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>>)) {
        nTuple<int, 3> i_lo, i_up;
        std::tie(i_lo, i_up) = src->as<std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>>>();
        SAMRAI::tbox::Dimension dim(3);
        dest->putDatabaseBox(uri, SAMRAI::tbox::DatabaseBox(dim, &(i_lo[0]), &(i_up[0])));
    }
    //    else if (src->type() == typeid(std::tuple<nTuple<Real, 3>, nTuple<Real, 3>>)) {
    //        nTuple<Real, 3> i_lo, i_up;
    //        std::tie(i_lo, i_up) = src->as<std::tuple<nTuple<Real, 3>, nTuple<Real, 3>>>();
    //        SAMRAI::tbox::Dimension dim(3);
    //        dest->putDatabaseBox(uri, SAMRAI::tbox::DatabaseBox(dim, &(i_lo[0]), &(i_up[0])));
    //    }
    else {
        WARNING << " Unknown type [" << src << "]" << std::endl;
    }
}
void DataBackendSAMRAI::pimpl_s::add_data_to_samrai(boost::shared_ptr<SAMRAI::tbox::Database>& lobj,
                                                    std::string const& uri,
                                                    std::shared_ptr<data::DataEntity> const& v) {
    UNSUPPORTED;
}
std::shared_ptr<DataEntity> DataBackendSAMRAI::pimpl_s::get_data_from_samrai(
    boost::shared_ptr<SAMRAI::tbox::Database> const& lobj) {}

std::shared_ptr<DataEntity> DataBackendSAMRAI::Get(std::string const& uri) const {
    pimpl_s::get_data_from_samrai(m_pimpl_->m_samrai_db_->getDatabase(uri));
}
std::shared_ptr<DataEntity> DataBackendSAMRAI::Get(id_type key) const {
    pimpl_s::get_data_from_samrai(m_pimpl_->m_samrai_db_->getDatabase(std::to_string(key)));
}
bool DataBackendSAMRAI::Set(std::string const& URI, std::shared_ptr<DataEntity> const& v) {
    pimpl_s::set_data_to_samrai(m_pimpl_->m_samrai_db_, URI, v);
}
bool DataBackendSAMRAI::Set(id_type key, std::shared_ptr<DataEntity> const& v) {
    pimpl_s::set_data_to_samrai(m_pimpl_->m_samrai_db_, std::to_string(key), v);
}
bool DataBackendSAMRAI::Add(std::string const& URI, std::shared_ptr<DataEntity> const& v) {
    pimpl_s::add_data_to_samrai(m_pimpl_->m_samrai_db_, URI, v);
}
bool DataBackendSAMRAI::Add(id_type key, std::shared_ptr<DataEntity> const& v) {
    pimpl_s::add_data_to_samrai(m_pimpl_->m_samrai_db_, std::to_string(key), v);
}
size_type DataBackendSAMRAI::Delete(std::string const& URI) { UNSUPPORTED; }
size_type DataBackendSAMRAI::Delete(id_type key) { UNSUPPORTED; }
void DataBackendSAMRAI::DeleteAll() { UNSUPPORTED; }

size_type DataBackendSAMRAI::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
    auto keys = m_pimpl_->m_samrai_db_->getAllKeys();
    for (auto const& k : keys) { fun(k, pimpl_s::get_data_from_samrai(m_pimpl_->m_samrai_db_->getDatabase(k))); }
}
size_type DataBackendSAMRAI::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const& fun) const {
    UNSUPPORTED;
    //    auto keys = m_pimpl_->m_samrai_db_->getAllKeys();
    //    for (int i = 0, ie = keys.size(); i < ie; ++i) {
    //        fun(keys[i], pimpl_s::get_data_from_samrai(m_pimpl_->m_samrai_db_->getDatabase(keys[i])));
    //    }
}

}  // namespace data{
}  // namespace simpla{