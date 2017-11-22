//
// Created by salmon on 17-9-24.
//
#include "HDF5Common.h"
#include <string>
namespace simpla {
namespace data {

hid_t GetHDF5DataType(std::type_info const& t_info) {
    hid_t v_type = H5T_NO_CLASS;

    if (t_info == typeid(char)) {
        v_type = H5T_NATIVE_CHAR;
    } else if (t_info == typeid(int)) {
        v_type = H5T_NATIVE_INT;
    } else if (t_info == typeid(long)) {
        v_type = H5T_NATIVE_LONG;
    } else if (t_info == typeid(unsigned int)) {
        v_type = H5T_NATIVE_UINT;
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
    // TODO: array data type
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

hid_t H5NumberType(std::type_info const& t_info) {
    hid_t v_type = H5T_NO_CLASS;
    if (t_info == typeid(char)) {
        v_type = H5T_NATIVE_CHAR;
    } else if (t_info == typeid(int)) {
        v_type = H5T_NATIVE_INT;
    } else if (t_info == typeid(long)) {
        v_type = H5T_NATIVE_LONG;
    } else if (t_info == typeid(unsigned int)) {
        v_type = H5T_NATIVE_UINT;
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
    return v_type;
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

void HDF5WriteArray(hid_t g_id, std::string const& key, std::shared_ptr<const ArrayBase> const& data) {
    bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
    //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
    H5O_info_t g_info;
    if (is_exist) { H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT)); }

    if (is_exist && g_info.type != H5O_TYPE_DATASET) {
        H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
        is_exist = false;
    }
    int ndims = data->GetNDIMS();

    index_type inner_lower[ndims];
    index_type inner_upper[ndims];
    index_type outer_lower[ndims];
    index_type outer_upper[ndims];

    data->GetIndexBox(inner_lower, inner_upper);
    data->GetShape(outer_lower, outer_upper);

    hsize_t m_shape[ndims];
    hsize_t m_start[ndims];
    hsize_t m_count[ndims];
    hsize_t m_stride[ndims];
    hsize_t m_block[ndims];

    if (data->isSlowFirst()) {
        for (int i = 0; i < ndims; ++i) {
            m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
            m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
            m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
            m_stride[i] = static_cast<hsize_t>(1);
            m_block[i] = static_cast<hsize_t>(1);
        }
    } else {
        for (int i = 0; i < ndims; ++i) {
            m_shape[ndims - 1 - i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
            m_start[ndims - 1 - i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
            m_count[ndims - 1 - i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
            m_stride[ndims - 1 - i] = static_cast<hsize_t>(1);
            m_block[ndims - 1 - i] = static_cast<hsize_t>(1);
        }
    }
    hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
    H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0], &m_block[0]));
    hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
    hid_t dset;
    hid_t d_type = GetHDF5DataType(data->value_type_info());
    H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, data->pointer()));

    H5_ERROR(H5Dclose(dset));
    if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
    if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
}
size_type HDF5SetEntity(hid_t g_id, std::string const& key, std::shared_ptr<const DataEntity> const& entity) {
    ASSERT(g_id > 0);

    size_type count = 0;
    if (key.empty()) {
    } else if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) > 0) {
        RUNTIME_ERROR << "Can not rewrite exist dataset/group! [" << key << "]" << std::endl;
    } else if (H5Aexists(g_id, key.c_str()) > 0) {
        H5_ERROR(H5Adelete(g_id, key.c_str()));
    }

    if (auto p = std::dynamic_pointer_cast<const DataLightT<std::string>>(entity)) {
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
        ++count;
    } else if (auto p = std::dynamic_pointer_cast<const DataLightT<std::string*>>(entity)) {
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
        count = s;
        //        FIXME << "Can not write string array to a HDF5 attribute!" << std::endl
        //              << "  key =" << key << " = " << *p << std::endl;
    } else if (auto p = std::dynamic_pointer_cast<const DataLight>(entity)) {
        hid_t d_type = -1;
        hid_t d_space;
        count = 1;
        auto ndims = p->rank();
        if (ndims > 0) {
            size_type d[ndims];
            hsize_t h5d[ndims];
            entity->extents(d);
            for (size_type i = 0; i < ndims; ++i) {
                h5d[i] = d[i];
                count *= d[i];
            }
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
    } else if (auto p = std::dynamic_pointer_cast<const ArrayBase>(entity)) {
        if (auto data = p->pointer()) {
            bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
            //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
            H5O_info_t g_info;
            if (is_exist) { H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT)); }

            if (is_exist && g_info.type != H5O_TYPE_DATASET) {
                H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
                is_exist = false;
            }
            static constexpr int ndims = 3;  // p->GetNDIMS();

            index_type inner_lower[ndims];
            index_type inner_upper[ndims];
            index_type outer_lower[ndims];
            index_type outer_upper[ndims];

            p->GetIndexBox(inner_lower, inner_upper);
            p->GetIndexBox(outer_lower, outer_upper);

            hsize_t m_shape[ndims];
            hsize_t m_start[ndims];
            hsize_t m_count[ndims];
            hsize_t m_stride[ndims];
            hsize_t m_block[ndims];
            for (int i = 0; i < ndims; ++i) {
                m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
                m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
                m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
                m_stride[i] = static_cast<hsize_t>(1);
                m_block[i] = static_cast<hsize_t>(1);
            }
            hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
            H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0], &m_block[0]));
            hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
            hid_t dset;
            hid_t d_type = GetHDF5DataType(p->value_type_info());
            H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, data));

            H5_ERROR(H5Dclose(dset));
            if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
            if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
            ++count;
        }
    }
    return count;
}

hid_t H5GroupTryOpen(hid_t root, std::string const& url) {
    if (url[0] == '/') { return H5GroupTryOpen(root, url.substr(1)); }

    hid_t gid = root;
    size_type begin = 0;
    size_type end = 0;
    while (begin != std::string::npos) {
        end = url.find('/', begin);
        auto tid = HDF5CreateOrOpenGroup(gid, url.substr(begin, end));
        if (gid != root) { H5Gclose(gid); }
        gid = tid;
        begin = end == std::string::npos ? std::string::npos : end + 1;
    };
    return gid;
}
hid_t HDF5CreateOrOpenGroup(hid_t grp, std::string const& key) {
    hid_t res;
    if (H5Lexists(grp, key.c_str(), H5P_DEFAULT) > 0) {
        H5O_info_t o_info;
        H5_ERROR(H5Oget_info_by_name(grp, key.c_str(), &o_info, H5P_DEFAULT));
        if (o_info.type == H5O_TYPE_GROUP) {
            res = H5Gopen(grp, key.c_str(), H5P_DEFAULT);
        } else {
            RUNTIME_ERROR << key << " is  a  dataset!";
        }
    } else if (H5Aexists(grp, key.c_str()) > 0) {
        RUNTIME_ERROR << key << " is  an attribute!";
    } else {
        res = H5Gcreate(grp, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    return res;
}
size_type HDF5Set(hid_t g_id, std::string const& key, std::shared_ptr<const DataEntry> node) {
    if (node == nullptr) { return 0; }

    size_type count = 0;
    switch (node->type()) {
        case DataEntry::DN_ARRAY:
        case DataEntry::DN_TABLE: {
            hid_t sub_gid = HDF5CreateOrOpenGroup(g_id, key);
            node->Foreach([&](std::string k, auto const& n) { count += HDF5Set(sub_gid, k, n); });
        } break;
        case DataEntry::DN_ENTITY:
            count = HDF5SetEntity(g_id, key, node->GetEntity());
            break;
        default:
            break;
    }
    return count;
}
size_type HDF5Add(hid_t g_id, std::string const& key, std::shared_ptr<const DataEntry> node) {
    if (node == nullptr) { return 0; }
    size_type count = 0;
    switch (node->type()) {
        case DataEntry::DN_ARRAY:
        case DataEntry::DN_TABLE: {
            hid_t sub_gid = HDF5CreateOrOpenGroup(g_id, key);
            node->Foreach([&](std::string k, auto const& n) { count += HDF5Set(sub_gid, k, n); });
        } break;
        case DataEntry::DN_ENTITY:
            count = HDF5SetEntity(g_id, key, node->GetEntity());
            break;
        default:
            break;
    }
    return count;
}
}
}