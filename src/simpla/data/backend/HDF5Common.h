//
// Created by salmon on 17-9-24.
//

#ifndef SIMPLA_H5COMMON_H
#define SIMPLA_H5COMMON_H

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
#define H5_ERROR(_FUN_)                                                               \
    if ((_FUN_) < 0) {                                                                \
        H5Eprint(H5E_DEFAULT, stderr);                                                \
        RUNTIME_ERROR << "\e[1;32m"                                                   \
                      << "HDF5 Error:" << __STRING(_FUN_) << "\e[1;37m" << std::endl; \
    }

hid_t H5NumberType(std::type_info const& t_info);

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

hid_t GetHDF5DataType(std::type_info const& t_info);

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

std::shared_ptr<DataEntity> HDF5GetEntity(hid_t obj_id, bool is_attribute);
size_type HDF5SetEntity(hid_t g_id, std::string const& key, std::shared_ptr<DataEntity> const& entity);
hid_t HDF5CreateOrOpenGroup(hid_t grp, std::string const& key);
hid_t H5GroupTryOpen(hid_t grp, std::string const& key);
size_type HDF5Set(hid_t g_id, std::string const& key, std::shared_ptr<DataNode> node);
size_type HDF5Add(hid_t g_id, std::string const& key, std::shared_ptr<DataNode> node);

void HDF5WriteArray(hid_t g_id, std::string const& key, std::shared_ptr<ArrayBase> const& data);
}
}

#endif  // SIMPLA_H5COMMON_H
