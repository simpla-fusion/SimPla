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

    std::shared_ptr<DataEntity> m_entity_ = nullptr;
    std::string m_key_;
};

DataNodeHDF5::DataNodeHDF5() : m_pimpl_(new pimpl_s) {}
DataNodeHDF5::DataNodeHDF5(pimpl_s* pimpl) : m_pimpl_(pimpl) {}
DataNodeHDF5::~DataNodeHDF5() {
    if (m_pimpl_->m_group_ > -1) { H5_ERROR(H5Gclose(m_pimpl_->m_group_)); }
    if (m_pimpl_->m_file_ > -1) { H5_ERROR(H5Fclose(m_pimpl_->m_file_)); }
    delete m_pimpl_;
}

int DataNodeHDF5::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    Disconnect();

    std::string filename =
        path.empty() ? "simpla_unnamed.h5" : path;  // = AutoIncreaseFileName(authority + "/" + path, "
    // .h5");

    LOGGER << "Create HDF5 File: [" << filename << "]" << std::endl;
    TODO << "Parser query : [ " << query << " ] and fragment : [ " << fragment << " ]" << std::endl;
    //    mkdir(authority.c_str(), 0777);
    H5_ERROR(m_pimpl_->m_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    m_pimpl_->m_key_ = "/";
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
    if (m_pimpl_->m_file_ != -1) {
        H5_ERROR(H5Fflush(m_pimpl_->m_file_, H5F_SCOPE_GLOBAL));
    } else if (m_pimpl_->m_group_ != -1) {
        H5_ERROR(H5Gflush(m_pimpl_->m_group_));
    }
    return SP_SUCCESS;
}
bool DataNodeHDF5::isValid() const { return m_pimpl_->m_group_ != -1; }

std::shared_ptr<DataNode> DataNodeHDF5::Duplicate() const {
    auto res = DataNodeHDF5::New();
    res->m_pimpl_->m_group_ = m_pimpl_->m_group_;
    res->m_pimpl_->m_entity_ = m_pimpl_->m_entity_;
    res->m_pimpl_->m_key_ = m_pimpl_->m_key_;
    return res;
}
size_type DataNodeHDF5::GetNumberOfChildren() const {
    size_type num = 0;
    if (m_pimpl_->m_group_ > 0) {
        H5G_info_t g_info;
        H5_ERROR(H5Gget_info(m_pimpl_->m_group_, &g_info));
        num += g_info.nlinks;
        H5O_info_t o_info;
        H5_ERROR(H5Oget_info(m_pimpl_->m_group_, &o_info));
        num += o_info.num_attrs;
    }
    return num;
}
DataNode::eNodeType DataNodeHDF5::NodeType() const {
    DataNode::eNodeType res = DN_NULL;
    auto num = GetNumberOfChildren();
    if (m_pimpl_->m_group_ != -1) {
        res = DN_TABLE;
    } else if (m_pimpl_->m_entity_ != nullptr || !m_pimpl_->m_key_.empty()) {
        res = DN_ENTITY;
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::Root() const { return Parent() != nullptr ? Parent()->Root() : Self(); }
std::shared_ptr<DataNode> DataNodeHDF5::Parent() const { return m_pimpl_->m_parent_; }

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
    //
    return (v_type);
}

template <typename U>
std::shared_ptr<DataEntity> HDF5GetEntityT(hid_t attr_id, hid_t d_type, hid_t d_space, bool is_attribute) {
    std::shared_ptr<DataEntity> res;
    if (is_attribute) {
        switch (H5Sget_simple_extent_type(d_space)) {
            case H5S_SCALAR: {
                U v;
                H5_ERROR(H5Aread(attr_id, d_type, &v));
                res = DataLightT<U>::New(v);
            } break;
            case H5S_SIMPLE: {
                int ndims = H5Sget_simple_extent_ndims(d_space);
                hsize_t dims[ndims];
                H5_ERROR(H5Sget_simple_extent_dims(d_space, dims, nullptr));
                std::shared_ptr<U> data(new U[H5Sget_simple_extent_npoints(d_space)]);
                H5_ERROR(H5Aread(attr_id, d_type, data.get()));
                res = DataLightT<U*>::New(ndims, dims, data);
            } break;
            case H5S_NULL:
            default:
                res = DataEntity::New();
        }
    }
    return res;
}

std::shared_ptr<DataEntity> HDF5GetEntity(hid_t obj_id, bool is_attribute) {
    std::shared_ptr<DataEntity> res = nullptr;
    hid_t d_type, d_space;
    if (is_attribute) {
        d_type = H5Aget_type(obj_id);
        d_space = H5Aget_space(obj_id);
    } else {
        d_type = H5Dget_type(obj_id);
        d_space = H5Dget_space(obj_id);
    }

    H5T_class_t type_class = H5Tget_class(d_type);

    if ((type_class == H5T_INTEGER || type_class == H5T_FLOAT)) {
        if (H5Tequal(d_type, H5T_NATIVE_HBOOL) > 0) {
            res = HDF5GetEntityT<bool>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_CHAR) > 0) {
            res = HDF5GetEntityT<char>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_SHORT) > 0) {
            res = HDF5GetEntityT<short>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_INT) > 0) {
            res = HDF5GetEntityT<int>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_LONG) > 0) {
            res = HDF5GetEntityT<double>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_LLONG) > 0) {
            res = HDF5GetEntityT<long long>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_UCHAR) > 0) {
            res = HDF5GetEntityT<unsigned char>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_USHORT) > 0) {
            res = HDF5GetEntityT<unsigned short>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_UINT) > 0) {
            res = HDF5GetEntityT<unsigned int>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULONG) > 0) {
            res = HDF5GetEntityT<unsigned long>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_ULLONG) > 0) {
            res = HDF5GetEntityT<unsigned long long>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_FLOAT) > 0) {
            res = HDF5GetEntityT<float>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_DOUBLE) > 0) {
            res = HDF5GetEntityT<double>(obj_id, d_type, d_space, is_attribute);
        } else if (H5Tequal(d_type, H5T_NATIVE_LDOUBLE) > 0) {
            res = HDF5GetEntityT<long double>(obj_id, d_type, d_space, is_attribute);
        }
    } else if (type_class == H5T_ARRAY) {
        FIXME;
    } else if (type_class == H5T_STRING && is_attribute) {
        switch (H5Sget_simple_extent_type(d_space)) {
            case H5S_SCALAR: {
                size_t sdims = H5Tget_size(d_type);
                char buffer[sdims + 1];
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, sdims));
                H5_ERROR(H5Aread(obj_id, m_type, buffer));
                H5_ERROR(H5Tclose(m_type));
                res = DataLightT<std::string>::New(std::string(buffer));
            } break;
            case H5S_SIMPLE: {
                hsize_t num = 0;
                H5_ERROR(H5Sget_simple_extent_dims(d_space, &num, nullptr));
                auto** buffer = new char*[num];
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
                H5_ERROR(H5Aread(obj_id, m_type, buffer));
                H5_ERROR(H5Tclose(m_type));
                auto p = DataLightT<std::string*>::New();
                for (int i = 0; i < num; ++i) {
                    p->value().push_back(std::string(buffer[i]));
                    delete buffer[i];
                }
                delete[] buffer;
                res = p;
            } break;
            default:
                break;
        }

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

std::shared_ptr<DataNodeHDF5> HDF5GetNode(hid_t grp, std::string const& uri) {
    auto res = DataNodeHDF5::New();
    res->m_pimpl_->m_key_ = uri;
    H5O_info_t o_info;

    if (H5Lexists(grp, uri.c_str(), H5P_DEFAULT) > 0 &&
        H5Oget_info_by_name(grp, uri.c_str(), &o_info, H5P_DEFAULT) >= 0) {
        switch (o_info.type) {
            case H5O_TYPE_GROUP:
                res->m_pimpl_->m_group_ = H5Gopen(grp, uri.c_str(), H5P_DEFAULT);
                break;
            case H5O_TYPE_DATASET: {
                auto d_id = H5Dopen(grp, uri.c_str(), H5P_DEFAULT);
                res->m_pimpl_->m_entity_ = HDF5GetEntity(d_id, false);
                H5_ERROR(H5Dclose(d_id));
            } break;
            default:
                RUNTIME_ERROR << "Undefined data type!" << std::endl;
                break;
        }
    } else if (H5Aexists(grp, uri.c_str()) != 0) {
        auto a_id = H5Aopen(grp, uri.c_str(), H5P_DEFAULT);
        res->m_pimpl_->m_entity_ = HDF5GetEntity(a_id, true);
        H5_ERROR(H5Aclose(a_id));
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(std::string const& uri, int flag) {
    std::shared_ptr<DataNodeHDF5> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeHDF5>(RecursiveFindNode(Self(), uri, flag).second);
    } else {
        if (m_pimpl_->m_group_ == -1 && (flag & NEW_IF_NOT_EXIST) != 0) {
            auto p_grp = (m_pimpl_->m_file_ != -1)
                             ? m_pimpl_->m_file_
                             : std::dynamic_pointer_cast<DataNodeHDF5>(Parent())->m_pimpl_->m_group_;

            if (H5Lexists(p_grp, m_pimpl_->m_key_.c_str(), H5P_DEFAULT) > 0) {
                m_pimpl_->m_group_ = H5Gopen(p_grp, m_pimpl_->m_key_.c_str(), H5P_DEFAULT);
                ASSERT(m_pimpl_->m_group_ > 0);
            } else {
                m_pimpl_->m_group_ = H5Gcreate(p_grp, m_pimpl_->m_key_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            };
        }
        if (m_pimpl_->m_group_ <= 0) {
            res = DataNodeHDF5::New();
            FIXME << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = HDF5GetNode(m_pimpl_->m_group_, uri);
            res->m_pimpl_->m_parent_ = Self();
        }
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(std::string const& uri, int flag) const {
    std::shared_ptr<DataNodeHDF5> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeHDF5>(RecursiveFindNode(Self(), uri, flag).second);
    } else {
        if (m_pimpl_->m_group_ <= 0) {
            RUNTIME_ERROR << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = HDF5GetNode(m_pimpl_->m_group_, uri);
            res->m_pimpl_->m_parent_ = Self();
        }
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(index_type s, int flag) { return GetNode(std::to_string(s), flag); }
std::shared_ptr<DataNode> DataNodeHDF5::GetNode(index_type s, int flag) const {
    return GetNode(std::to_string(s), flag);
}
size_type DataNodeHDF5::DeleteNode(std::string const& uri, int flag) {
    size_type count = 0;
    if ((flag & RECURSIVE) == 0 && m_pimpl_->m_group_ != -1) {
        if (H5Aexists(m_pimpl_->m_group_, uri.c_str()) > 0) {
            H5Adelete(m_pimpl_->m_group_, uri.c_str());
            ++count;
        } else if (H5Lexists(m_pimpl_->m_group_, uri.c_str(), H5P_DEFAULT) != 0) {
            H5_ERROR(H5Ldelete(m_pimpl_->m_group_, uri.c_str(), H5P_DEFAULT));
            ++count;
        }
    } else {
        auto r = RecursiveFindNode(Self(), uri, RECURSIVE);
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

std::shared_ptr<DataEntity> DataNodeHDF5::GetEntity() const {
    return (m_pimpl_->m_entity_ != nullptr) ? m_pimpl_->m_entity_ : DataEntity::New();
}

int HDF5Set(hid_t g_id, std::string const& key, std::shared_ptr<DataEntity> const& entity) {
    ASSERT(g_id > 0);

    int count = 0;

    if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) > 0) {
        RUNTIME_ERROR << "Can not rewrite exist dataset/group!" << std::endl;
    } else if (H5Aexists(g_id, key.c_str()) > 0) {
        H5_ERROR(H5Adelete(g_id, key.c_str()));
    }

    if (auto p = std::dynamic_pointer_cast<DataLightT<std::string>>(entity)) {
        std::string s_str = p->value();
        auto m_type = H5Tcopy(H5T_C_S1);
        H5_ERROR(H5Tset_size(m_type, s_str.size()));
        H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
        auto m_space = H5Screate(H5S_SCALAR);
        auto aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        H5_ERROR(H5Awrite(aid, m_type, s_str.c_str()));
        H5_ERROR(H5Tclose(m_type));
        H5_ERROR(H5Sclose(m_space));
        H5_ERROR(H5Aclose(aid));
    } else if (auto p = std::dynamic_pointer_cast<DataLightT<std::string*>>(entity)) {
        std::vector<char const*> s_array;
        for (auto const& v : p->value()) { s_array.push_back(v.c_str()); }
        hsize_t s = s_array.size();
        auto m_space = H5Screate_simple(1, &s, nullptr);
        auto m_type = H5Tcopy(H5T_C_S1);
        H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
        auto aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
        H5_ERROR(H5Awrite(aid, m_type, &s_array[0]));
        H5_ERROR(H5Tclose(m_type));
        H5_ERROR(H5Sclose(m_space));
        H5_ERROR(H5Aclose(aid));
        //        FIXME << "Can not write string array to a HDF5 attribute!" << std::endl
        //              << "  key =" << key << " = " << *p << std::endl;
    } else if (auto p = std::dynamic_pointer_cast<DataLight>(entity)) {
        hid_t d_type = -1;
        hid_t d_space;
        auto ndims = p->rank();
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
            auto aid = H5Acreate(g_id, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT);

            if (p->isContinue()) {
                H5_ERROR(H5Awrite(aid, d_type, p->GetPointer()));

            } else {
                auto* ptr = operator new(p->GetAlignOf());
                p->CopyOut(ptr);
                H5_ERROR(H5Awrite(aid, d_type, ptr));
                operator delete(ptr);
            }
            H5_ERROR(H5Aclose(aid));

        } else {
            FIXME << "Can not write hdf5 attribute! " << std::endl << key << " = " << *entity << std::endl;
        }
        H5_ERROR(H5Sclose(d_space));

        ++count;
    } else if (auto p = std::dynamic_pointer_cast<DataBlock>(entity)) {
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
        //    H5_ERROR(dset = H5Dcreate(loc_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT,
        //    H5P_DEFAULT));
        //    H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, src->GetPointer()));
        //
        //    H5_ERROR(H5Dclose(dset));
        //    if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
        //    if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
        ++count;
    }
    return count;
}

size_type DataNodeHDF5::SetEntity(std::shared_ptr<DataEntity> const& v) {
    size_type res = 0;
    auto parent = std::dynamic_pointer_cast<DataNodeHDF5>(Parent());
    if (parent == nullptr || m_pimpl_->m_key_.empty()) {
        FIXME << "Can not set value to group node or unnamed node [ " << m_pimpl_->m_key_ << " ]" << std::endl;
    } else {
        res = HDF5Set(parent->m_pimpl_->m_group_, m_pimpl_->m_key_, v);
    }
    return res;
}
size_type DataNodeHDF5::AddEntity(std::shared_ptr<DataEntity> const& v) { return AddNode()->SetEntity(v); }

size_type DataNodeHDF5::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) {
    if (m_pimpl_->m_group_ == -1) { return 0; };
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_pimpl_->m_group_, &g_info));

    size_type count = 0;
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num =
            H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                           H5P_DEFAULT);

        count += fun(std::string(buffer), GetNode(std::string(buffer), 0));
    }
    H5O_info_t o_info;
    H5_ERROR(H5Oget_info(m_pimpl_->m_group_, &o_info));
    for (hsize_t i = 0; i < o_info.num_attrs; ++i) {
        ssize_t num =
            H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5_ERROR(H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer,
                                    static_cast<size_t>(num + 1), H5P_DEFAULT));

        count += fun(std::string(buffer), GetNode(std::string(buffer), 0));
    }
    return count;
}
size_type DataNodeHDF5::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    if (m_pimpl_->m_group_ == -1) { return 0; };
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_pimpl_->m_group_, &g_info));

    size_type count = 0;
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num =
            H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5Lget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                           H5P_DEFAULT);

        count += fun(std::string(buffer), GetNode(std::string(buffer), 0));
    }
    H5O_info_t o_info;
    H5_ERROR(H5Oget_info(m_pimpl_->m_group_, &o_info));
    for (hsize_t i = 0; i < o_info.num_attrs; ++i) {
        ssize_t num =
            H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5_ERROR(H5Aget_name_by_idx(m_pimpl_->m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer,
                                    static_cast<size_t>(num + 1), H5P_DEFAULT));

        count += fun(std::string(buffer), GetNode(std::string(buffer), 0));
    }
    return count;
}

}  // namespace data{
}  // namespace simpla{