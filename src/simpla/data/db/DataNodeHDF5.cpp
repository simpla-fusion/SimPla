//
// Created by salmon on 17-3-10.
//
#include "DataNodeHDF5.h"
#include <sys/stat.h>
#include <regex>

#include "../DataBlock.h"
#include "../DataEntity.h"
#include "../DataNode.h"

#include "simpla/parallel/MPIComm.h"
extern "C" {
#include <hdf5.h>
#include <hdf5_hl.h>
}

namespace simpla {
namespace data {
REGISTER_CREATOR(DataNodeHDF5, h5);

#define H5_ERROR(_FUN_)                                                               \
    if ((_FUN_) < 0) {                                                                \
        H5Eprint(H5E_DEFAULT, stderr);                                                \
        RUNTIME_ERROR << "\e[1;32m"                                                   \
                      << "HDF5 Error:" << __STRING(_FUN_) << "\e[1;37m" << std::endl; \
    }

struct DataNodeHDF5::pimpl_s {
    std::shared_ptr<DataNodeHDF5> m_parent_;

    hid_t m_file_ = -1;
    hid_t m_group_ = -1;
    hid_t m_attr_ = -1;
    hid_t m_dataset_ = -1;

    std::string m_key_;
    int m_idx_ = -1;
};

DataNodeHDF5::DataNodeHDF5() : m_pimpl_(new pimpl_s) {}
DataNodeHDF5::DataNodeHDF5(pimpl_s* pimpl) : m_pimpl_(pimpl) {}
DataNodeHDF5::~DataNodeHDF5() {
    if (m_pimpl_->m_attr_ > -1) { H5_ERROR(H5Aclose(m_pimpl_->m_attr_)); }
    if (m_pimpl_->m_dataset_ > -1) { H5_ERROR(H5Dclose(m_pimpl_->m_dataset_)); }
    if (m_pimpl_->m_group_ > -1) { H5_ERROR(H5Gclose(m_pimpl_->m_group_)); }
    if (m_pimpl_->m_file_ > -1) { H5_ERROR(H5Fclose(m_pimpl_->m_file_)); }
    delete m_pimpl_;
}

int DataNodeHDF5::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    Disconnect();

    std::string filename = path;  // = AutoIncreaseFileName(authority + "/" + path, ".h5");

    LOGGER << "Create HDF5 File: [" << filename << "]" << std::endl;

    //    mkdir(authority.c_str(), 0777);
    H5_ERROR(m_pimpl_->m_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    H5_ERROR(m_pimpl_->m_group_ = H5Gopen(m_pimpl_->m_file_, "/", H5P_DEFAULT));
    return SP_SUCCESS;
};
int DataNodeHDF5::Disconnect() {
    auto root = std::dynamic_pointer_cast<DataNodeHDF5>(Root());
    if (root->m_pimpl_->m_file_ > -1) { H5Fclose(root->m_pimpl_->m_file_); }
    root->m_pimpl_->m_file_ = -1;
    root->m_pimpl_->m_group_ = -1;
    return SP_SUCCESS;
}
int DataNodeHDF5::Flush() {
    auto root = std::dynamic_pointer_cast<DataNodeHDF5>(Root());
    if (m_pimpl_->m_group_ != -1) {
        H5_ERROR(H5Gflush(m_pimpl_->m_group_));
    } else {
        Parent()->Flush();
    }
    return SP_SUCCESS;
}
bool DataNodeHDF5::isValid() const { return m_pimpl_->m_group_ != -1; }

std::shared_ptr<DataNode> DataNodeHDF5::Duplicate() const {
    auto res = DataNodeHDF5::New();
    //    res->m_pimpl_->m_file_ = m_pimpl_->m_file_;
    res->m_pimpl_->m_group_ = m_pimpl_->m_group_;
    res->m_pimpl_->m_attr_ = m_pimpl_->m_attr_;
    res->m_pimpl_->m_dataset_ = m_pimpl_->m_dataset_;
    res->m_pimpl_->m_key_ = m_pimpl_->m_key_;
    res->m_pimpl_->m_idx_ = m_pimpl_->m_idx_;
    return res;
}
size_type DataNodeHDF5::GetNumberOfChildren() const { return 0; }
DataNode::e_NodeType DataNodeHDF5::NodeType() const {
    DataNode::e_NodeType res = DN_NULL;
    auto num = GetNumberOfChildren();
    if (m_pimpl_->m_group_ != -1) {
        res = DN_TABLE;
    } else if (m_pimpl_->m_dataset_ != -1 || m_pimpl_->m_attr_ != -1 || !m_pimpl_->m_key_.empty()) {
        res = DN_ENTITY;
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::Root() const {
    return Parent() != nullptr ? Parent()->Root() : const_cast<this_type*>(this)->shared_from_this();
}
std::shared_ptr<DataNode> DataNodeHDF5::Parent() const { return m_pimpl_->m_parent_; }

template <typename U>
std::shared_ptr<DataEntity> read_attr(hid_t attr_id, hid_t d_type) {
    U res;
    H5Aread(attr_id, d_type, &res);
    return make_data(res);
}

template <template <typename> class TFun, typename... Args>
void H5TypeDispatch(hid_t d_type, Args&&... args) {
    H5T_class_t type_class = H5Tget_class(d_type);

    if ((type_class == H5T_INTEGER || type_class == H5T_FLOAT)) {
        if (H5Tequal(d_type, H5T_NATIVE_CHAR) > 0) {
            TFun<char>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_SHORT) > 0) {
            TFun<short>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_INT) > 0) {
            TFun<int>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_LONG) > 0) {
            TFun<double>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_LLONG) > 0) {
            TFun<long long>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_UCHAR) > 0) {
            TFun<unsigned char>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_USHORT) > 0) {
            TFun<unsigned short>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_UINT) > 0) {
            TFun<unsigned int>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULONG) > 0) {
            TFun<unsigned long>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULLONG) > 0) {
            TFun<unsigned long long>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_FLOAT) > 0) {
            TFun<float>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_DOUBLE) > 0) {
            TFun<double>(std::forward<Args>(args)...);
        } else if (H5Tequal(d_type, H5T_NATIVE_LDOUBLE) > 0) {
            TFun<long double>(std::forward<Args>(args)...);
        }
    } else if (type_class == H5T_ARRAY) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_STRING) {
        TFun<std::string>(std::forward<Args>(args)...);
    } else if (type_class == H5T_TIME) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_BITFIELD) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_REFERENCE) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_ENUM) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_VLEN) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_NO_CLASS) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_OPAQUE) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_COMPOUND) {
        UNIMPLEMENTED;
    }
}
std::shared_ptr<DataBlock> HDF5GetDataSet(hid_t ds_id) {
    std::shared_ptr<DataBlock> res = nullptr;
    return res;
}
std::shared_ptr<DataEntity> HDF5GetAttr(hid_t attr_id) {
    std::shared_ptr<DataEntity> res = nullptr;
    hid_t d_type = H5Aget_type(attr_id);
    hid_t d_space = H5Aget_space(attr_id);

    H5T_class_t type_class = H5Tget_class(d_type);

    if ((type_class == H5T_INTEGER || type_class == H5T_FLOAT)) {
        if (H5Tequal(d_type, H5T_NATIVE_CHAR) > 0) {
            res = read_attr<char>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_SHORT) > 0) {
            res = read_attr<short>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_INT) > 0) {
            res = read_attr<int>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_LONG) > 0) {
            res = read_attr<double>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_LLONG) > 0) {
            res = read_attr<long long>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_UCHAR) > 0) {
            res = read_attr<unsigned char>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_USHORT) > 0) {
            res = read_attr<unsigned short>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_UINT) > 0) {
            res = read_attr<unsigned int>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULONG) > 0) {
            res = read_attr<unsigned long>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULLONG) > 0) {
            res = read_attr<unsigned long long>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_FLOAT) > 0) {
            res = read_attr<float>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_DOUBLE) > 0) {
            res = read_attr<double>(attr_id, d_type);
        } else if (H5Tequal(d_type, H5T_NATIVE_LDOUBLE) > 0) {
            res = read_attr<long double>(attr_id, d_type);
        }
    } else if (type_class == H5T_ARRAY) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_STRING) {
        size_t sdims = H5Tget_size(d_type);
        ++sdims;
        char buffer[sdims];
        hid_t m_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(m_type, sdims);
        H5Aread(attr_id, m_type, buffer);
        res = make_data<std::string>(std::string(buffer));
        H5Tclose(m_type);
    } else if (type_class == H5T_TIME) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_BITFIELD) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_REFERENCE) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_ENUM) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_VLEN) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_NO_CLASS) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_OPAQUE) {
        UNIMPLEMENTED;
    } else if (type_class == H5T_COMPOUND) {
        UNIMPLEMENTED;
    }

    H5Tclose(d_type);
    H5Sclose(d_space);
    return res;
}
hid_t GetHDF5DataType(std::type_info const& t_info) {
    hid_t v_type = H5T_NO_CLASS;

    if (t_info == typeid(int)) {
        v_type = H5T_NATIVE_INT;
    } else if (t_info == typeid(long)) {
        v_type = H5T_NATIVE_LONG;
    } else if (t_info == typeid(unsigned long)) {
        v_type = H5T_NATIVE_ULONG;
    } else if (t_info == typeid(float)) {
        v_type = H5T_NATIVE_FLOAT;
    } else if (t_info == typeid(double)) {
        v_type = H5T_NATIVE_DOUBLE;
    } else if (t_info == typeid(std::complex<double>)) {
        H5_ERROR(v_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
        H5_ERROR(H5Tinsert(v_type, "r", 0, H5T_NATIVE_DOUBLE));
        H5_ERROR(H5Tinsert(v_type, "i", sizeof(double), H5T_NATIVE_DOUBLE));

    }
    // TODO:
    //   else if (d_type->isArray()) {
    //        auto const& t_array = d_type->cast_as<DataArray>();
    //        hsize_t dims[t_array.rank()];
    //        for (int i = 0; i < t_array.rank(); ++i) { dims[i] = t_array.dimensions()[i]; }
    //        hid_t res2 = res;
    //        H5_ERROR(res2 = H5Tarray_create(res, t_array.rank(), dims));
    //        if (H5Tcommitted(res) > 0) H5_ERROR(H5Tclose(res));
    //        res = res2;
    //    } else if (d_type->isTable()) {
    //        H5_ERROR(v_type = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte()));
    //
    //        for (auto const& item : d_type.members()) {
    //            hid_t t_member = convert_data_type_sp_to_h5(std::get<0>(item), true);
    //
    //            H5_ERROR(H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item), t_member));
    //            if (H5Tcommitted(t_member) > 0) H5_ERROR(H5Tclose(t_member));
    //        }
    //    }
    else {
        RUNTIME_ERROR << "Unknown m_data type:" << t_info.name();
    }

    return (v_type);
}

int HDF5GetNode(DataNodeHDF5* res, hid_t g_id, std::string const& key) {
    H5O_info_t g_info;
    if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) == 0) {
        res->m_pimpl_->m_key_ = key;
    } else if (H5Oexists_by_name(g_id, key.c_str(), H5P_DEFAULT) > 0 &&
               H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT) >= 0 && g_info.type == H5O_TYPE_GROUP) {
        res->m_pimpl_->m_group_ = H5Gopen(g_id, key.c_str(), H5P_DEFAULT);
    } else {
        res->m_pimpl_->m_key_ = key;
    }
    return SP_SUCCESS;
}

std::shared_ptr<DataEntity> HDF5Get(hid_t g_id, std::string const& key) {
    std::shared_ptr<DataEntity> res = nullptr;
    ASSERT(g_id > -1);
    H5O_info_t g_info;
    if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) == 0) {
    } else if (H5Oexists_by_name(g_id, key.c_str(), H5P_DEFAULT) > 0 &&
               H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT) >= 0 && g_info.type == H5O_TYPE_DATASET) {
        auto ds_id = H5Dopen(g_id, key.c_str(), H5P_DEFAULT);
        res = HDF5GetDataSet(ds_id);
        H5_ERROR(H5Aclose(ds_id));
    } else if (H5Aexists(g_id, key.c_str()) != 0) {
        auto aid = H5Aopen(g_id, key.c_str(), H5P_DEFAULT);
        res = HDF5GetAttr(aid);
        H5_ERROR(H5Aclose(aid));
    }
    //    else {
    //        RUNTIME_ERROR << "Object type is  not  DataSet/Attribute!" << std::endl;
    //    }
    return res == nullptr ? DataEntity::New() : res;
}

int HDF5Set(hid_t g_id, std::string const& key, std::shared_ptr<DataEntity> const& entity) {
    ASSERT(g_id != -1);
    if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) > 0) { H5_ERROR(H5Ldelete(g_id, key.c_str(), H5P_DEFAULT)); }
    if (H5Aexists(g_id, key.c_str()) > 0) { H5_ERROR(H5Adelete(g_id, key.c_str())); }
    int count = 0;
    hid_t aid;
    if (auto p = std::dynamic_pointer_cast<DataLightT<std::string>>(entity)) {
        std::string const& s_str = p->as<std::string>();
        hid_t m_type = H5Tcopy(H5T_C_S1);
        H5_ERROR(H5Tset_size(m_type, s_str.size()));
        H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
        hid_t m_space = H5Screate(H5S_SCALAR);
        aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        H5_ERROR(H5Awrite(aid, m_type, s_str.c_str()));
        H5_ERROR(H5Tclose(m_type));
    } else if (auto p = std::dynamic_pointer_cast<DataLight>(entity)) {
        hid_t d_type = -1;
        hid_t d_space;
        auto ndims = static_cast<int>(entity->rank());
        if (ndims > 0) {
            size_type d[ndims];
            hsize_t h5d[ndims];
            entity->extents(d);
            for (size_type i = 0; i < ndims; ++i) { h5d[i] = d[i]; }
            d_space = H5Screate_simple(ndims, h5d, nullptr);
        } else {
            d_space = H5Screate(H5S_SCALAR);
        }

        if (false) {}
#define DEC_TYPE(_T_, _H5_T_)                            \
    else if (entity->value_type_info() == typeid(_T_)) { \
        d_type = _H5_T_;                                 \
    }

        DEC_TYPE(bool, H5T_NATIVE_HBOOL)
        DEC_TYPE(float, H5T_NATIVE_FLOAT)
        DEC_TYPE(double, H5T_NATIVE_DOUBLE)
        DEC_TYPE(int, H5T_NATIVE_INT)
        DEC_TYPE(long, H5T_NATIVE_LONG)
        DEC_TYPE(unsigned int, H5T_NATIVE_UINT)
        DEC_TYPE(unsigned long, H5T_NATIVE_ULONG)
#undef DEC_TYPE

        if (d_type != -1) {
            aid = H5Acreate(g_id, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT);
            H5_ERROR(H5Awrite(aid, d_type, p->GetPointer()));
        } else {
            WARNING << "Unsupported  data type! [" << entity->value_type_info().name() << "]" << std::endl;
        }
        H5_ERROR(H5Sclose(d_space));
        H5_ERROR(H5Aclose(aid));
        ++count;
    } else if (auto p = std::dynamic_pointer_cast<DataBlock>(entity)) {
        //        ++count;
    }
    return count;
}

// int HDF5Set(hid_t* ds_id, hid_t g_id, std::string const& key, std::shared_ptr<DataBlock> const& src) {
//    if (src->isEmpty()) {
//        LOGGER << "Write Empty DataBlock" << key << std::endl;
//        return 0;
//    }
//    bool is_exist = H5Lexists(loc_id, key.c_str(), H5P_DEFAULT) != 0;
//    //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
//    H5O_info_t g_info;
//    if (is_exist) { H5_ERROR(H5Oget_info_by_name(loc_id, key.c_str(), &g_info, H5P_DEFAULT)); }
//    if (is_exist && !overwrite) { return 0; }
//
//    if (overwrite && is_exist && g_info.type != H5O_TYPE_DATASET) {
//        H5Ldelete(loc_id, key.c_str(), H5P_DEFAULT);
//        is_exist = false;
//    }
//    const int ndims = src->GetNDIMS();
//
//    index_type inner_lower[ndims];
//    index_type inner_upper[ndims];
//    index_type outer_lower[ndims];
//    index_type outer_upper[ndims];
//
//    src->GetIndexBox(inner_lower, inner_upper);
//    src->GetIndexBox(outer_lower, outer_upper);
//
//    hsize_t m_shape[ndims];
//    hsize_t m_start[ndims];
//    hsize_t m_count[ndims];
//    hsize_t m_stride[ndims];
//    hsize_t m_block[ndims];
//    for (int i = 0; i < ndims; ++i) {
//        m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
//        m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
//        m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
//        m_stride[i] = static_cast<hsize_t>(1);
//        m_block[i] = static_cast<hsize_t>(1);
//    }
//    hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
//    H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0],
//    &m_block[0]));
//    hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
//    hid_t dset;
//    hid_t d_type = GetHDF5DataType(src->value_type_info());
//    H5_ERROR(dset = H5Dcreate(loc_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
//    H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, src->GetPointer()));
//
//    H5_ERROR(H5Dclose(dset));
//    if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
//    if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
//
//    return 0;
//}

std::shared_ptr<DataNode> DataNodeHDF5::GetNode(std::string const& uri, int flag) {
    std::shared_ptr<DataNodeHDF5> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeHDF5>(
            RecursiveFindNode(const_cast<this_type*>(this)->shared_from_this(), uri, flag).second);
    } else {
        res = DataNodeHDF5::New();
        res->m_pimpl_->m_parent_ =
            std::dynamic_pointer_cast<DataNodeHDF5>(const_cast<this_type*>(this)->shared_from_this());
        if (m_pimpl_->m_group_ == -1 && (flag & NEW_IF_NOT_EXIST) != 0) {
            auto parent = std::dynamic_pointer_cast<DataNodeHDF5>(Parent());
            if (parent != nullptr && parent->m_pimpl_->m_group_ != -1) {
                m_pimpl_->m_group_ = H5Gcreate(parent->m_pimpl_->m_group_, m_pimpl_->m_key_.c_str(), H5P_DEFAULT,
                                               H5P_DEFAULT, H5P_DEFAULT);
            }
        }
        if (m_pimpl_->m_group_ == -1) { RUNTIME_ERROR << "Can not get sub-node from non-group object!" << std::endl; }
        HDF5GetNode(res.get(), m_pimpl_->m_group_, uri);
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(std::string const& uri, int flag) const {
    std::shared_ptr<DataNodeHDF5> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeHDF5>(
            RecursiveFindNode(const_cast<this_type*>(this)->shared_from_this(), uri, flag).second);
    } else {
        if (m_pimpl_->m_group_ == -1) { RUNTIME_ERROR << "Can not get sub-node from non-group object!" << std::endl; }
        res = DataNodeHDF5::New(new DataNodeHDF5::pimpl_s);
        HDF5GetNode(res.get(), m_pimpl_->m_group_, uri);
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(index_type s, int flag) { return GetNode(std::to_string(s), flag); }
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(index_type s, int flag) const {
    return GetNode(std::to_string(s), flag);
}
int DataNodeHDF5::DeleteNode(std::string const& uri, int flag) {
    int count = 0;
    if ((flag & RECURSIVE) == 0 && m_pimpl_->m_group_ != -1) {
        if (H5Aexists(m_pimpl_->m_group_, uri.c_str()) > 0) {
            H5Adelete(m_pimpl_->m_group_, uri.c_str());
            ++count;
        } else if (H5Lexists(m_pimpl_->m_group_, uri.c_str(), H5P_DEFAULT) != 0) {
            H5_ERROR(H5Ldelete(m_pimpl_->m_group_, uri.c_str(), H5P_DEFAULT));
            ++count;
        }
    } else {
        auto r = RecursiveFindNode(shared_from_this(), uri, RECURSIVE);
        if (r.second != nullptr && r.second->Parent() != nullptr) {
            r.second->Parent()->DeleteNode(r.first, 0);
            ++count;
        }
    }

    return count;
}
void DataNodeHDF5::Clear() {
    if (m_pimpl_->m_group_ != -1) {
        // TODO: delete all;
    }
}

std::shared_ptr<DataEntity> DataNodeHDF5::Get() const {
    auto parent = std::dynamic_pointer_cast<DataNodeHDF5>(Parent());
    ASSERT(parent != nullptr);
    return HDF5Get(m_pimpl_->m_parent_->m_pimpl_->m_group_, m_pimpl_->m_key_);
}
int DataNodeHDF5::Set(std::shared_ptr<DataEntity> const& v) {
    int res = 0;
    auto parent = std::dynamic_pointer_cast<DataNodeHDF5>(Parent());
    if (m_pimpl_->m_group_ != -1 || m_pimpl_->m_key_.empty() || parent == nullptr) {
        RUNTIME_ERROR << "Can not set value to group node or unnamed node " << std::endl;
    } else {
        res = HDF5Set(parent->m_pimpl_->m_group_, m_pimpl_->m_key_, v);
    }
    return res;
}
int DataNodeHDF5::Add(std::shared_ptr<DataEntity> const& v) { return AddNode()->Set(v); }

// size_type DataBaseHDF5::Foreach(
//    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
//    auto res = get_table_from_h5(m_pack_->m_group_, uri, false);
//    if (res.first == -1 || res.second.empty()) { return 0; }
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
//    op.m_op_(std::string(attr_name), DataBaseHDF5::pack_s::HDF5Get(location_id, std::string(attr_name)));
//}
int DataNodeHDF5::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) {
    if (m_pimpl_->m_group_ == -1) { return 0; };
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_pimpl_->m_group_, &g_info));

    int count = 0;
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num =
            H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                           H5P_DEFAULT);
        auto node = DataNodeHDF5::New();
        node->m_pimpl_->m_parent_ = std::dynamic_pointer_cast<DataNodeHDF5>(this->shared_from_this());
        node->m_pimpl_->m_key_ = std::string(buffer);
        count += fun(std::string(buffer), node);
    }
    H5O_info_t o_info;
    H5_ERROR(H5Oget_info(m_pimpl_->m_group_, &o_info));
    for (hsize_t i = 0; i < o_info.num_attrs; ++i) {
        ssize_t num =
            H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5_ERROR(H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer,
                                    static_cast<size_t>(num + 1), H5P_DEFAULT));
        auto node = DataNodeHDF5::New();
        node->m_pimpl_->m_parent_ = std::dynamic_pointer_cast<DataNodeHDF5>(this->shared_from_this());
        node->m_pimpl_->m_key_ = std::string(buffer);
        count += fun(std::string(buffer), node);
    }
    return count;
}
//        int DataNodeHDF5::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {}
int DataNodeHDF5::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    if (m_pimpl_->m_group_ == -1) { return 0; };
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_pimpl_->m_group_, &g_info));

    int count = 0;
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num =
            H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                           H5P_DEFAULT);
        auto node = DataNodeHDF5::New();
        node->m_pimpl_->m_parent_ =
            std::dynamic_pointer_cast<DataNodeHDF5>(const_cast<DataNodeHDF5*>(this)->shared_from_this());
        node->m_pimpl_->m_key_ = std::string(buffer);
        count += fun(std::string(buffer), node);
    }
    H5O_info_t o_info;
    H5_ERROR(H5Oget_info(m_pimpl_->m_group_, &o_info));
    for (hsize_t i = 0; i < o_info.num_attrs; ++i) {
        ssize_t num =
            H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5_ERROR(H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer,
                                    static_cast<size_t>(num + 1), H5P_DEFAULT));
        auto node = DataNodeHDF5::New();
        node->m_pimpl_->m_parent_ =
            std::dynamic_pointer_cast<DataNodeHDF5>(const_cast<DataNodeHDF5*>(this)->shared_from_this());
        node->m_pimpl_->m_key_ = std::string(buffer);
        count += fun(std::string(buffer), node);
    }
    return count;
}

}  // namespace data{
}  // namespace simpla{