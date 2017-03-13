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

// template <typename U, int NDIMS>
// class DataEntityHeavyArrayHDF5 : public DataEntityWrapper<Array<U, NDIMS>> {};
struct DataSourceHDF5 {
    hid_t m_hdf5_;
};
template <typename U>
struct DataEntityHeavyHDF5 : public DataEntity {};

struct H5GroupCloser {
    H5GroupCloser() {}
    ~H5GroupCloser() {}
    inline void operator()(void* ptr) {
        hid_t* f = reinterpret_cast<hid_t*>(ptr);
        if (*f != -1) { H5Gclose(*f); }
        delete f;
    }
};
struct DataBackendHDF5::pimpl_s {
    std::shared_ptr<DataSourceHDF5> m_data_src_ = nullptr;
    hid_t m_f_id_ = -1;
    hid_t m_g_id_ = -1;

    static std::pair<std::shared_ptr<hid_t>, std::string> get_table(hid_t self, std::string const& uri,
                                                                    bool create_if_need = false);

    static std::regex sub_dir_regex;
};
DataBackendHDF5::DataBackendHDF5() : m_pimpl_(new pimpl_s) {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5 const& other) : DataBackendHDF5() {
    m_pimpl_->m_data_src_ = other.m_pimpl_->m_data_src_;
}
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
    if (m_pimpl_->m_f_id_ != -1) {
        H5Fclose(m_pimpl_->m_f_id_);
        m_pimpl_->m_f_id_ = -1;
    }
};
std::ostream& DataBackendHDF5::Print(std::ostream& os, int indent) const { return os; }
std::shared_ptr<DataBackend> DataBackendHDF5::Duplicate() const { return std::make_shared<DataBackendHDF5>(*this); }
std::shared_ptr<DataBackend> DataBackendHDF5::CreateNew() const { return std::make_shared<DataBackendHDF5>(); }

void DataBackendHDF5::Flush() { H5Fflush(m_pimpl_->m_data_src_->m_hdf5_, H5F_SCOPE_GLOBAL); }
bool DataBackendHDF5::isNull() const { return m_pimpl_->m_data_src_ == nullptr; }
size_type DataBackendHDF5::size() const { UNIMPLEMENTED; }

std::regex DataBackendHDF5::pimpl_s::sub_dir_regex(R"(([^/?#:]+)/)", std::regex::extended | std::regex::optimize);

std::pair<std::shared_ptr<hid_t>, std::string> DataBackendHDF5::pimpl_s::get_table(hid_t self, std::string const& uri,
                                                                                   bool create_if_need) {
    std::smatch sub_dir_match_result;
    hid_t gid = self;

    auto pos = uri.begin();
    auto end = uri.end();
    int count = 0;

    for (; std::regex_search(pos, end, sub_dir_match_result, sub_dir_regex);
         pos = sub_dir_match_result.suffix().first) {
        auto path = sub_dir_match_result.str(1);
        hid_t g_id2;
        // H5Gget_objinfo(gid, path.c_str(), 0, NULL) == 0
        if (H5Lexists(gid, path.c_str(), H5P_DEFAULT) == 0) {
            H5_ERROR(g_id2 = H5Gcreate(gid, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        } else {
            H5_ERROR(g_id2 = H5Gopen(gid, path.c_str(), H5P_DEFAULT));

            if (g_id2 == -1) {
                RUNTIME_ERROR << std::endl
                              << std::setw(25) << std::right << "illegal path [/" << uri << "]" << std::endl
                              << std::setw(25) << std::right << "     at here   "
                              << std::setw(&(*pos) - &(*uri.begin())) << " "
                              << " ^" << std::endl;
            }
        }
        if (count > 0 && gid > 0) { H5Gclose(gid); }
        gid = g_id2;
        ++count;
    }

    std::shared_ptr<hid_t> p(new hid_t, H5GroupCloser());
    *p = gid;
    return std::make_pair(p, std::string(""));

    //    bool success = false;
    //
    //    std::smatch sub_dir_match_result;
    //
    //    auto pos = uri.begin();
    //    auto end = uri.end();
    //    std::shared_ptr<DataTable> result(nullptr);
    //    for (; std::regex_search(pos, end, sub_dir_match_result, sub_group_regex);
    //         pos = sub_dir_match_result.suffix().first) {
    //        auto k = sub_dir_match_result.str(1);
    //
    //        if (t->isDatabase(k)) {
    //            t = t->getDatabase(k);
    //        } else if (!t->isDatabase(k) && create_if_need) {
    //            t = t->putDatabase(k);
    //        } else {
    //            RUNTIME_ERROR << std::endl
    //                          << std::setw(25) << std::right << "illegal path [/" << uri << "]" << std::endl
    //                          << std::setw(25) << std::right << "     at here   " << std::setw(&(*pos) -
    //                          &(*uri.begin()))
    //                          << " "
    //                          << " ^" << std::endl;
    //        }
    //    }
};
std::shared_ptr<DataEntity> DataBackendHDF5::Get(std::string const& uri) const {
    auto res = m_pimpl_->get_table(m_pimpl_->m_g_id_, uri);
    return nullptr;
}
void DataBackendHDF5::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    auto res = m_pimpl_->get_table(m_pimpl_->m_f_id_, uri);
}
void DataBackendHDF5::Add(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
size_type DataBackendHDF5::Delete(std::string const& URI) { UNIMPLEMENTED; }

size_type DataBackendHDF5::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}

}  // namespace data{
}  // namespace simpla{