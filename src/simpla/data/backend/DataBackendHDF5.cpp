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
//   public:
//    HDF5Closer() {}
//    ~HDF5Closer() {}
//    inline void
//};
void HDF5Closer(void* ptr) {
    hid_t f = *reinterpret_cast<hid_t*>(ptr);
    if (f != -1) { H5Fclose(f); }
}
struct DataBackendHDF5::pimpl_s {
    std::shared_ptr<hid_t> m_f_id_;
    hid_t m_g_id_ = -1;

    static std::shared_ptr<DataEntity> HDF5Get(DataBackendHDF5* self, hid_t loc_id, std::string const& name);
};

static std::pair<hid_t, std::string> get_table_from_h5(hid_t root, std::string const& uri,
                                                       bool return_if_not_exist = false) {
    return HierarchicalGetTable(
        root, uri, [&](hid_t g, std::string const& k) { return H5Lexists(g, k.c_str(), H5P_DEFAULT) != 0; },
        [&](hid_t g, std::string const& k) { return H5Gopen(g, k.c_str(), H5P_DEFAULT); },
        [&](hid_t g, std::string const& k) { return H5Gcreate(g, k.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); });
};

std::shared_ptr<DataEntity> convert_data_from_h5_attr(hid_t loc_id, std::string const& name) {
    hid_t attr_id = -1;
    H5_ERROR(attr_id = H5Aopen(loc_id, name.c_str(), H5P_DEFAULT));
    hid_t d_type = H5Aget_type(attr_id);

    std::shared_ptr<DataEntity> res;
    bool bad_cast_error = false;
    H5T_class_t type_class = H5Tget_class(d_type);

    if (type_class == H5T_INTEGER || type_class == H5T_FLOAT || type_class == H5T_ARRAY || type_class == H5T_STRING) {
        hid_t d_space = H5Aget_space(attr_id);
        char buffer[H5Aget_storage_size(attr_id)];
        H5Aread(attr_id, d_type, buffer);

        hid_t atomic_id = H5Tget_native_type(type_class, H5T_DIR_ASCEND);

        if (atomic_id == H5T_NATIVE_CHAR) {
            res = make_data_entity(*reinterpret_cast<char const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_SHORT) {
            res = make_data_entity(*reinterpret_cast<short const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_INT) {
            res = make_data_entity(*reinterpret_cast<int const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_LONG) {
            res = make_data_entity(*reinterpret_cast<double const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_LLONG) {
            res = make_data_entity(*reinterpret_cast<long long const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_UCHAR) {
            res = make_data_entity(*reinterpret_cast<unsigned char const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_USHORT) {
            res = make_data_entity(*reinterpret_cast<unsigned short const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_UINT) {
            res = make_data_entity(*reinterpret_cast<unsigned int const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_ULONG) {
            res = make_data_entity(*reinterpret_cast<unsigned long const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_ULLONG) {
            res = make_data_entity(*reinterpret_cast<unsigned long long const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_FLOAT) {
            res = make_data_entity(*reinterpret_cast<float const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_DOUBLE) {
            res = make_data_entity(*reinterpret_cast<double const*>(buffer));
        } else if (atomic_id == H5T_NATIVE_LDOUBLE) {
            res = make_data_entity(*reinterpret_cast<long double const*>(buffer));
        } else if (atomic_id == H5T_C_S1)

            H5_ERROR(H5Tclose(atomic_id));

        H5Tclose(d_type);
        H5Sclose(d_space);
        if (attr_id != -1) { H5Aclose(attr_id); }
    }

    //    else if (type_class == H5T_STRING) {
    //        auto size = H5Tget_size(d_type);
    //
    //        htri_t size_var;
    //
    //
    //
    //    }
    //    else if (type_class == H5T_NO_CLASS) {
    //        bad_cast_error = true;
    //    }
    //    else if (type_class == H5T_OPAQUE) {
    //        data::DataType(std::type_index(typeid(void)), H5Tget_size(t_id)).swap(dtype);
    //    } else if (type_class == H5T_COMPOUND) {
    //        for (int i = 0, num = H5Tget_nmembers(t_id); i < num; ++i) {
    //            dtype.push_back(convert_data_type_h5_to_sp(H5Tget_member_type(t_id, i)),
    //                            std::string(H5Tget_member_name(t_id, i)), static_cast<int>(H5Tget_member_offset(t_id,
    //                            i)));
    //        }
    //
    //    } else if (type_class == H5T_TIME) {
    //        UNIMPLEMENTED;
    //    } else if (type_class == H5T_STRING) {
    //        UNIMPLEMENTED;
    //    } else if (type_class == H5T_BITFIELD) {
    //        UNIMPLEMENTED;
    //    } else if (type_class == H5T_REFERENCE) {
    //        UNIMPLEMENTED;
    //    } else if (type_class == H5T_ENUM) {
    //        UNIMPLEMENTED;
    //    } else if (type_class == H5T_VLEN) {
    //        UNIMPLEMENTED;
    //    }

    //    if (type_class == H5T_ARRAY) {
    //        int rank = H5Tget_array_ndims(t_id);
    //        hsize_t dims[rank];
    //        size_type dims2[rank];
    //        for (int i = 0; i < rank; ++i) { dims2[i] = (size_type)dims[i]; }
    //        H5_ERROR(H5Tget_array_dims(t_id, dims));
    //
    //        dtype.extent(rank, dims2);
    //    }
}
std::shared_ptr<DataEntity> DataBackendHDF5::pimpl_s::HDF5Get(DataBackendHDF5* self, hid_t loc_id,
                                                              std::string const& name) {
    if (H5Lexists(loc_id, name.c_str(), H5P_DEFAULT) == 0) { return std::make_shared<DataEntity>(); }
    std::shared_ptr<DataEntity> res;
    if (H5Oexists_by_name(loc_id, name.c_str(), H5P_DEFAULT) != 0) {
        H5O_info_t g_info;
        H5_ERROR(H5Oget_info_by_name(loc_id, name.c_str(), &g_info, H5P_DEFAULT));
        switch (g_info.type) {
            case H5O_TYPE_GROUP: {
                auto t_backend = std::make_shared<DataBackendHDF5>();
                t_backend->m_pimpl_->m_f_id_ = self->m_pimpl_->m_f_id_;
                t_backend->m_pimpl_->m_g_id_ = H5Gopen(loc_id, name.c_str(), H5P_DEFAULT);
                res = std::make_shared<DataTable>(t_backend);
                break;
            }
            case H5O_TYPE_DATASET:
                UNIMPLEMENTED;
                break;
            default:
                break;
        }
    } else if (H5Aexists(loc_id, name.c_str())) {
        res = convert_data_from_h5_attr(loc_id, name);
    }
    return std::make_shared<DataEntity>();
}

void HDF5Set(hid_t g_id, std::string const& key, std::shared_ptr<DataEntity> const& src, bool do_add = false) {
    if (src->isTable()) {
        hid_t sub_gid = H5Gcreate(g_id, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        src->cast_as<DataTable>().Accept(
            [&](std::string const& k, std::shared_ptr<data::DataEntity> const& v) { HDF5Set(sub_gid, k, v); });
        H5Gclose(sub_gid);
        return;
    } else if (src->isHeavyBlock()) {
        return;
    } else if (src->type() == typeid(std::string)) {
        std::string const& s_str = data_cast<std::string>(*src);
        hid_t m_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(m_type, s_str.size());
        H5Tset_strpad(m_type, H5T_STR_NULLTERM);
        hid_t m_space = H5Screate(H5S_SCALAR);
        hid_t a_id = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a_id, m_type, s_str.c_str());
        H5Tclose(m_type);
        H5Aclose(a_id);
    } else {
        hid_t d_type;
        hid_t d_space;
        char* data = nullptr;
        if (src->isArray()) {
            hsize_t s = src->size();
            d_space = H5Screate_simple(1, &s, NULL);
        } else {
            d_space = H5Screate(H5S_SCALAR);
        }

        if (false) {}
#define DEC_TYPE(_T_, _H5_T_)                                                                 \
    else if (src->type() == typeid(_T_)) {                                                    \
        d_type = _H5_T_;                                                                      \
        if (src->isArray()) {                                                                 \
            data = reinterpret_cast<char*>(&src->cast_as<DataArrayWrapper<_T_>>().data()[0]); \
        } else {                                                                              \
            data = new char[sizeof(_T_)];                                                     \
            *reinterpret_cast<_T_*>(data) = data_cast<_T_>(*src);                             \
        }                                                                                     \
    }

        //        DEC_TYPE(bool, H5T_NATIVE_HBOOL)
        DEC_TYPE(float, H5T_NATIVE_FLOAT)
        DEC_TYPE(double, H5T_NATIVE_DOUBLE)
        DEC_TYPE(int, H5T_NATIVE_INT)
        DEC_TYPE(long, H5T_NATIVE_LONG)
        DEC_TYPE(unsigned int, H5T_NATIVE_UINT)
        DEC_TYPE(unsigned long, H5T_NATIVE_ULONG)
#undef DEC_TYPE

        hid_t a_id;
        if (do_add) {
            return;
        } else {
            if (H5Aexists(g_id, key.c_str())) { H5Adelete(g_id, key.c_str()); }
            a_id = H5Acreate(g_id, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT);
        }
        H5Awrite(a_id, d_type, data);
        H5Aclose(a_id);

        if (!src->isArray()) { delete data; }
    }
}

DataBackendHDF5::DataBackendHDF5() : m_pimpl_(new pimpl_s) {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5 const& other) : DataBackendHDF5() {
    m_pimpl_->m_f_id_ = other.m_pimpl_->m_f_id_;
    m_pimpl_->m_g_id_ = other.m_pimpl_->m_g_id_;
}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendHDF5::DataBackendHDF5(std::string const& uri, std::string const& status) : DataBackendHDF5() {
    Connect(uri, status);
}
DataBackendHDF5::~DataBackendHDF5() {}

void DataBackendHDF5::Connect(std::string const& path, std::string const& param) {
    Disconnect();
    //    m_pimpl_->m_f_id_ = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    m_pimpl_->m_f_id_ = std::shared_ptr<hid_t>(new hid_t, HDF5Closer);

    //    if (GLOBAL_COMM.num_of_process() > 1) {
    //        hid_t plist_id;
    //
    //        H5_ERROR(plist_id = H5Pcreate(H5P_FILE_ACCESS));
    //        H5_ERROR(H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info()));
    //        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id));
    //        H5_ERROR(H5Pclose(plist_id));
    //    }
    LOGGER << "Create HDF5 File: " << path << std::endl;

    H5_ERROR(*m_pimpl_->m_f_id_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    H5_ERROR(m_pimpl_->m_g_id_ = H5Gopen(*m_pimpl_->m_f_id_, "/", H5P_DEFAULT));
};
void DataBackendHDF5::Disconnect() {
    if (m_pimpl_->m_g_id_ != -1) {
        H5Fclose(m_pimpl_->m_g_id_);
        m_pimpl_->m_g_id_ = -1;
    }
    m_pimpl_->m_f_id_.reset();
};
std::ostream& DataBackendHDF5::Print(std::ostream& os, int indent) const { return os; }
std::shared_ptr<DataBackend> DataBackendHDF5::Duplicate() const { return std::make_shared<DataBackendHDF5>(*this); }
std::shared_ptr<DataBackend> DataBackendHDF5::CreateNew() const { return std::make_shared<DataBackendHDF5>(); }

void DataBackendHDF5::Flush() { H5Fflush(*m_pimpl_->m_f_id_, H5F_SCOPE_GLOBAL); }
bool DataBackendHDF5::isNull() const { return m_pimpl_->m_f_id_ == nullptr; }
size_type DataBackendHDF5::size() const { UNIMPLEMENTED; }

std::shared_ptr<DataEntity> DataBackendHDF5::Get(std::string const& uri) const {
    auto res = get_table_from_h5(m_pimpl_->m_g_id_, uri, true);
    return (res.first == -1) ? nullptr : pimpl_s::HDF5Get(const_cast<DataBackendHDF5*>(this), res.first, res.second);
}
void DataBackendHDF5::Set(std::string const& uri, std::shared_ptr<DataEntity> const& src) {
    if (src == nullptr) { return; }
    auto res = get_table_from_h5(m_pimpl_->m_g_id_, uri, false);
    if (res.first == -1 || res.second == "") { return; }
    HDF5Set(res.first, res.second, src, false);
}
void DataBackendHDF5::Add(std::string const& uri, std::shared_ptr<DataEntity> const& src) {
    if (src == nullptr) { return; }
    auto res = get_table_from_h5(m_pimpl_->m_g_id_, uri, false);
    if (res.first == -1 || res.second == "") { return; }
    HDF5Set(res.first, res.second, src, true);
}
size_type DataBackendHDF5::Delete(std::string const& uri) {
    auto res = get_table_from_h5(*m_pimpl_->m_f_id_, uri, false);
    if (res.first == -1 || res.second == "") { return 0; }
    if (H5Aexists(res.first, res.second.c_str())) { H5Adelete(res.first, res.second.c_str()); }
}

//size_type DataBackendHDF5::Accept(
//    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
//    auto res = get_table_from_h5(m_pimpl_->m_g_id_, uri, false);
//    if (res.first == -1 || res.second == "") { return 0; }
//    if (H5Aexists(res.first, res.second.c_str())) {
//        H5_ERROR(H5Adelete(res.first, res.second.c_str()));
//        return 1;
//    } else {
//        return 0;
//    }
//}
//
// struct attr_op {
//    std::function<void(std::string, std::shared_ptr<DataEntity>)> m_op_;
//};
// herr_t attr_info(hid_t location_id /*in*/, const char* attr_name /*in*/, const H5A_info_t* ainfo /*in*/,
//                 void* op_data /*in,out*/) {
//    auto const& op = *reinterpret_cast<attr_op*>(op_data);
//    op.m_op_(std::string(attr_name), DataBackendHDF5::pimpl_s::HDF5Get(location_id, std::string(attr_name)));
//}
size_type DataBackendHDF5::Accept(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
    //    H5_ERROR(H5Aiterate2(m_pimpl_->m_g_id_, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, attr_info, &op));
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_pimpl_->m_g_id_, &g_info));

    char dot[] = ".";
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num =
            H5Lget_name_by_idx(m_pimpl_->m_g_id_, dot, H5_INDEX_NAME, H5_ITER_NATIVE, i, NULL, 0, H5P_DEFAULT);
        char buffer[num];
        H5_ERROR(
            H5Lget_name_by_idx(m_pimpl_->m_g_id_, dot, H5_INDEX_NAME, H5_ITER_NATIVE, i, buffer, num, H5P_DEFAULT));
        fun(std::string(buffer), std::make_shared<DataEntity>());
    }
    return g_info.nlinks;
 }

}  // namespace data{
}  // namespace simpla{