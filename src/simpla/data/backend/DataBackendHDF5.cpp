//
// Created by salmon on 17-3-10.
//
#include "DataBackendHDF5.h"
#include <simpla/algebra/Array.h>
#include <simpla/algebra/nTuple.h>
#include <regex>

#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
extern "C" {
#include <hdf5.h>
#include <hdf5_hl.h>
}

#include <H5FDmpio.h>
#include <simpla/data/DataUtility.h>

namespace simpla {
namespace data {

#define H5_ERROR(_FUN_)                                                               \
    if ((_FUN_) < 0) {                                                                \
        H5Eprint(H5E_DEFAULT, stderr);                                                \
        RUNTIME_ERROR << "\e[1;32m"                                                   \
                      << "HDF5 Error:" << __STRING(_FUN_) << "\e[1;37m" << std::endl; \
    }

// struct HDF5Status {
//    HDF5Status(std::string const& url, std::string const& status = "");
//    ~HDF5Status();
//    hid_t base_file_id_ = -1;
//    hid_t base_group_id_ = -1;
//    void Close();
//    void Open(std::string const& url, std::string const& status);
//};
// HDF5Status::HDF5Status(std::string const& url, std::string const& status) { Open(url, status); }
// HDF5Status::~HDF5Status() {
//    if (base_file_id_ != -1) { Close(); }
//}
// class HDF5Closer {
//    void* addr_;
//    size_t s_;
//    HDF5Closer(void* p, size_t s) : addr_(p), s_(s) {}
//    ~HDF5Closer() {}
//    inline void operator()(void* ptr) {
//        hid_t f = *reinterpret_cast<hid_t*>(ptr);
//        if (f != -1) { H5Fclose(f); }
//    }
//};

struct DataBackendHDF5::pimpl_s {
    hid_t m_f_id_;
    hid_t m_g_id_;
};

static std::pair<hid_t, std::string> get_table_from_h5(hid_t root, std::string const& uri,
                                                       bool return_if_not_exist = false) {
    return HierarchicalGetTable(
        root, uri, [&](hid_t g, std::string const& k) { return H5Lexists(g, k.c_str(), H5P_DEFAULT) != 0; },
        [&](hid_t g, std::string const& k) { return H5Gopen(g, k.c_str(), H5P_DEFAULT); },
        [&](hid_t g, std::string const& k) { return H5Gcreate(g, k.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); });
};

DataBackendHDF5::DataBackendHDF5() : m_pimpl_(new pimpl_s) {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5 const& other) : DataBackendHDF5() {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendHDF5::DataBackendHDF5(std::string const& uri, std::string const& status) : DataBackendHDF5() {
    Connect(uri, status);
}
DataBackendHDF5::~DataBackendHDF5() {}

void DataBackendHDF5::Connect(std::string const& path, std::string const& param) {
    Disconnect();
    //    m_pimpl_->m_f_id_ = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    hid_t f_id;

    //    if (GLOBAL_COMM.num_of_process() > 1) {
    //        hid_t plist_id;
    //
    //        H5_ERROR(plist_id = H5Pcreate(H5P_FILE_ACCESS));
    //        H5_ERROR(H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info()));
    //        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id));
    //        H5_ERROR(H5Pclose(plist_id));
    //    }
    LOGGER << "Create HDF5 File: " << path << std::endl;
    { H5_ERROR(f_id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)); }

    m_pimpl_->m_f_id_ = f_id;
};
void DataBackendHDF5::Disconnect() {
    if (m_pimpl_->m_g_id_ != -1) {
        H5Fclose(m_pimpl_->m_g_id_);
        m_pimpl_->m_g_id_ = -1;
    } else if (m_pimpl_->m_f_id_ != -1) {
        H5Fclose(m_pimpl_->m_f_id_);
        m_pimpl_->m_f_id_ = -1;
    }
};
std::ostream& DataBackendHDF5::Print(std::ostream& os, int indent) const { return os; }
std::shared_ptr<DataBackend> DataBackendHDF5::Duplicate() const { return std::make_shared<DataBackendHDF5>(*this); }
std::shared_ptr<DataBackend> DataBackendHDF5::CreateNew() const { return std::make_shared<DataBackendHDF5>(); }

void DataBackendHDF5::Flush() { H5Fflush(m_pimpl_->m_f_id_, H5F_SCOPE_GLOBAL); }
bool DataBackendHDF5::isNull() const { return m_pimpl_->m_f_id_ == -1; }
size_type DataBackendHDF5::size() const { UNIMPLEMENTED; }

std::shared_ptr<DataEntity> DataBackendHDF5::Get(std::string const& uri) const {
    auto res = get_table_from_h5(m_pimpl_->m_g_id_, uri, true);
    return nullptr;
}
void DataBackendHDF5::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    if (v == nullptr) { return; }
    if (!v->isLight()) {
        UNIMPLEMENTED;
        return;
    }
    auto res = get_table_from_h5(m_pimpl_->m_f_id_, uri, false);
    if (res.first == -1 || res.second == "") { return; }
    if (v->type() == typeid(std::string)) {
        std::string const& s_str = data_cast<std::string>(*v);
        hid_t m_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(m_type, s_str.size());
        H5Tset_strpad(m_type, H5T_STR_NULLTERM);
        hid_t m_space = H5Screate(H5S_SCALAR);
        hid_t a_id = H5Acreate(res.first, res.second.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a_id, m_type, s_str.c_str());
        H5Tclose(m_type);
        H5Aclose(a_id);
    } else {
        UNIMPLEMENTED;
        //        hid_t m_type = convert_data_type_sp_to_h5(any_v.data_type());
        //
        //        hid_t m_space = H5Screate(H5S_SCALAR);
        //
        //        hid_t a_id = H5Acreate(loc_id, GetName.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        //
        //        H5Awrite(a_id, m_type, any_v.m_data());
        //
        //        if (H5Tcommitted(m_type) > 0)
        //        {
        //            H5Tclose(m_type);
        //        }
        //
        //        H5Aclose(a_id);
        //
        //        H5Sclose(m_space);
    }
}
void DataBackendHDF5::Add(std::string const& uri, std::shared_ptr<DataEntity> const&) {
    auto res = get_table_from_h5(m_pimpl_->m_f_id_, uri, false);
}
size_type DataBackendHDF5::Delete(std::string const& uri) {
    UNIMPLEMENTED;
    //    auto res = get_table_from_h5(m_pimpl_->m_f_id_, uri, true);
}

size_type DataBackendHDF5::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}

}  // namespace data{
}  // namespace simpla{