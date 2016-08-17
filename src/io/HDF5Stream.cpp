/**
 * @file data_stream.cpp
 *
 *  created on: 2013-12-12
 *      Author: salmon
 */

extern "C"
{
#include <hdf5.h>
#include <hdf5_hl.h>
}

#include <H5FDmpio.h>
#include <cstring> //for memcopy

#include "HDF5Stream.h"
#include "../data_model/DataSet.h"


#include "../parallel/Parallel.h"
#include "../parallel/MPIComm.h"
#include "../parallel/MPIAuxFunctions.h"


#include "../gtl/MiscUtilities.h"
#include "../gtl/MemoryPool.h"

#define H5_ERROR(_FUN_) if((_FUN_)<0){H5Eprint(H5E_DEFAULT, stderr); \
RUNTIME_ERROR<<"\e[1;32m" <<"HDF5 Error:" <<__STRING(_FUN_) <<  "\e[1;37m"<<std::endl;}

//THROW_EXCEPTION_RUNTIME_ERROR("\e[1;32m" ,"HDF5 Error:",__STRING(_FUN_),  "\e[1;37m","");

namespace simpla { namespace io
{
hid_t convert_data_type_sp_to_h5(data_model::DataType const &, size_t flag = 0UL);

hid_t convert_data_space_sp_to_h5(data_model::DataSpace const &, size_t flag = 0UL);

data_model::DataType convert_data_type_h5_to_sp(hid_t);

data_model::DataSpace convert_data_space_h5_to_sp(hid_t);

struct HDF5Stream::pimpl_s
{

    pimpl_s();

    ~pimpl_s();

    hid_t base_file_id_ = -1;
    hid_t base_group_id_ = -1;

    void set_attribute(hid_t loc_id, std::string const &name, any const &v);

    void set_attribute(hid_t loc_id, Properties const &v);

    Properties get_attribute(hid_t loc_id, std::string const &name) const;

    Properties get_attribute(hid_t loc_id, int idx = -1) const;

    static constexpr size_t DEFAULT_MAX_BUFFER_DEPTH = 100;

    std::map<std::string, data_model::DataSet> m_buffer_map_;

};

HDF5Stream::HDF5Stream()
    : m_pimpl_(new pimpl_s) {}

HDF5Stream::~HDF5Stream() { close(); }

bool HDF5Stream::is_valid() const
{
    return m_pimpl_ != nullptr && m_pimpl_->base_file_id_ > 0;
}

bool HDF5Stream::is_opened() const
{
    return m_pimpl_ != nullptr && m_pimpl_->base_file_id_ > 0;
}

std::string HDF5Stream::absolute_path(std::string const &url) const
{
    std::string file_name;
    std::string grp_name;
    std::string obj_name;

    if (url != "")
    {
        std::tie(file_name, grp_name, obj_name, std::ignore) = IOStream::parser_url(url);
    }
    if (file_name == "") { file_name = IOStream::current_file_name(); }
    if (grp_name == "") { grp_name = IOStream::current_group_name(); }
    return file_name + ":" + grp_name + obj_name;
}

std::tuple<bool, std::string> HDF5Stream::open(std::string const &url, int flag)
{
    std::string file_name = IOStream::current_file_name();
    std::string grp_name = IOStream::current_group_name();
    std::string obj_name = "";

    if (url != "")
    {
        std::tie(file_name, grp_name, obj_name, std::ignore) = IOStream::parser_url(url);
    }

    //TODO using regex parser url

    if (IOStream::current_file_name() != file_name || m_pimpl_->base_file_id_ <= 0)
    {
        open_file(file_name, flag);
    }
    if (IOStream::current_group_name() != grp_name) { close_group(); }

    if (m_pimpl_->base_group_id_ <= 0) { open_group(grp_name); }

    if (obj_name != "" && ((flag & (SP_APPEND | SP_RECORD)) == 0UL))
    {
        if (GLOBAL_COMM.process_num() == 0)
        {
            obj_name =
                obj_name +

                    AutoIncrease([&](std::string const &s) -> bool
                                 {
                                     return H5Lexists(m_pimpl_->base_group_id_,
                                                      (obj_name + s).c_str(), H5P_DEFAULT) > 0;
                                 }, 0, 4);
        }

        parallel::bcast_string(&obj_name);
    }

    bool is_existed = false;

    if (obj_name != "")
    {
        is_existed = H5Lexists(m_pimpl_->base_group_id_, obj_name.c_str(), H5P_DEFAULT) != 0;
    }

    return std::make_tuple(is_existed, obj_name);
}

void HDF5Stream::set_attribute(std::string const &url, Properties const &any_v)
{

    delete_attribute(url);

//    data_model::DataType dtype = any_v.data_type();
//
//    void const *v = any_v.m_data();
//
//    std::string file_name, grp_path, obj_name, attr_name;
//
//    std::tie(file_name, grp_path, obj_name, attr_name) = IOStream::parser_url(url);
//
//    hid_t g_id, o_id;
    UNIMPLEMENTED;
// FIXME
//    std::tie(grp_path, g_id) = m_self_->open_group(grp_path);
//
//    if (o_id != g_id) { H5Oclose(o_id); }
//
//    if (g_id != m_self_->base_group_id_) { H5Gclose(g_id); }

}

void HDF5Stream::pimpl_s::set_attribute(hid_t loc_id, std::string const &name,
                                        any const &any_v)
{

    if (any_v.is_same<std::string>())
    {
        std::string const &s_str = any_v.as<std::string>();

        hid_t m_type = H5Tcopy(H5T_C_S1);

        H5Tset_size(m_type, s_str.size());

        H5Tset_strpad(m_type, H5T_STR_NULLTERM);

        hid_t m_space = H5Screate(H5S_SCALAR);

        hid_t a_id = H5Acreate(loc_id, name.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);

        H5Awrite(a_id, m_type, s_str.c_str());

        H5Tclose(m_type);

        H5Aclose(a_id);
    }
    else
    {
        UNIMPLEMENTED;
//        hid_t m_type = convert_data_type_sp_to_h5(any_v.data_type());
//
//        hid_t m_space = H5Screate(H5S_SCALAR);
//
//        hid_t a_id = H5Acreate(loc_id, name.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
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

void HDF5Stream::pimpl_s::set_attribute(hid_t loc_id, Properties const &prop)
{
    for (auto const &item : prop)
    {
        set_attribute(loc_id, item.first, item.second);
    }

}

Properties HDF5Stream::pimpl_s::get_attribute(hid_t loc_id, std::string const &name) const
{

    UNIMPLEMENTED;
    return std::move(Properties());
}

Properties HDF5Stream::pimpl_s::get_attribute(hid_t loc_id, int idx) const
{
    UNIMPLEMENTED;
    Properties res;

    return std::move(res);
}

Properties HDF5Stream::get_attribute(std::string const &url) const
{
    UNIMPLEMENTED;
    return std::move(Properties());
}

void HDF5Stream::delete_attribute(std::string const &url)
{
    std::string file_name, grp_name, obj_name, attr_name;

    std::tie(file_name, grp_name, obj_name, attr_name) = IOStream::parser_url(url);

//    if (obj_name != "")
//    {
    // FIXME UNIMPLEMENTED;
//        hid_t g_id;
//        std::tie(grp_name, g_id) = m_self_->open_group(grp_name);
//
//        if (H5Aexists_by_name(g_id, obj_name.c_str(), attr_name.c_str(), H5P_DEFAULT))
//        {
//            H5Adelete_by_name(g_id, obj_name.c_str(), attr_name.c_str(), H5P_DEFAULT);
//        }
//        if (g_id != m_self_->base_group_id_)
//        {
//            H5Gclose(g_id);
//        }
//    }
    UNIMPLEMENTED;

}

HDF5Stream::pimpl_s::pimpl_s()
{
    base_file_id_ = -1;
    base_group_id_ = -1;

    hid_t error_stack = H5Eget_current_stack();
    H5Eset_auto(error_stack, NULL, NULL);
}

HDF5Stream::pimpl_s::~pimpl_s()
{
}

void HDF5Stream::close_group()
{
    if (m_pimpl_ == nullptr) { return; }
    if (m_pimpl_->base_group_id_ > 0)
    {
        H5_ERROR(H5Gclose(m_pimpl_->base_group_id_));
        m_pimpl_->base_group_id_ = -1;
    }

}

void HDF5Stream::close_file()
{
    if (m_pimpl_ == nullptr) { return; }

    close_group();

    if (m_pimpl_->base_file_id_ > 0)
    {
        flush();
        H5_ERROR(H5Fclose(m_pimpl_->base_file_id_));
        m_pimpl_->base_file_id_ = -1;
        VERBOSE << "File [" << IOStream::current_file_name() << "] is closed!" << std::endl;
    }

}

void HDF5Stream::close()
{
    close_file();
}

void HDF5Stream::open_file(std::string const &fname, bool is_append)
{

    std::string filename = is_append ? fname : IOStream::auto_increase_file_name(fname);

//    if (filename != IOStream::current_file_name()) { close_file(); }

    hid_t f_id;

    if (GLOBAL_COMM.num_of_process() > 1)
    {
        hid_t plist_id;

        H5_ERROR(plist_id = H5Pcreate(H5P_FILE_ACCESS));

        H5_ERROR(H5Pset_fapl_mpio(plist_id, GLOBAL_COMM.comm(), GLOBAL_COMM.info()));

        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id));

        H5_ERROR(H5Pclose(plist_id));

    }
    else
    {
        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    }


    m_pimpl_->base_file_id_ = f_id;

    IOStream::current_file_name(filename);

    VERBOSE << "File [" << filename << "] is opened!" << std::endl;

}

void HDF5Stream::open_group(std::string const &str)
{
    std::string path = str;
    hid_t g_id = -1;

    if (path[0] != '/')
    {
        path = IOStream::current_group_name() + path;
    }

    if (path[path.size() - 1] != '/')
    {
        path = path + "/";
    }

    if (path == "/" || H5Lexists(m_pimpl_->base_file_id_, path.c_str(), H5P_DEFAULT) != 0)
    {
        H5_ERROR(g_id = H5Gopen(m_pimpl_->base_file_id_, path.c_str(), H5P_DEFAULT));
    }
    else
    {
        H5_ERROR(g_id = H5Gcreate(m_pimpl_->base_file_id_, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }
    m_pimpl_->base_group_id_ = g_id;

    IOStream::current_group_name(path);

}

hid_t convert_data_type_sp_to_h5(data_model::DataType const &d_type, size_t is_compact_array)
{
    hid_t res = H5T_NO_CLASS;

    if (!d_type.is_valid()) THROW_EXCEPTION_RUNTIME_ERROR("illegal m_data type");

    if (!d_type.is_compound())
    {

        if (d_type.template is_same<int>())
        {
            res = H5T_NATIVE_INT;
        }
        else if (d_type.template is_same<long>())
        {
            res = H5T_NATIVE_LONG;
        }
        else if (d_type.template is_same<unsigned long>())
        {
            res = H5T_NATIVE_ULONG;
        }
        else if (d_type.template is_same<float>())
        {
            res = H5T_NATIVE_FLOAT;
        }
        else if (d_type.template is_same<double>())
        {
            res = H5T_NATIVE_DOUBLE;
        }
        else if (d_type.template is_same<std::complex<double>>())
        {
            H5_ERROR(res = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
            H5_ERROR(H5Tinsert(res, "r", 0, H5T_NATIVE_DOUBLE));
            H5_ERROR(H5Tinsert(res, "i", sizeof(double), H5T_NATIVE_DOUBLE));

        }
        else
        {
            RUNTIME_ERROR << "Unknown m_data type:" << d_type.name();
        }

        if (d_type.is_array())
        {
            hsize_t dims[d_type.rank()];

            for (int i = 0; i < d_type.rank(); ++i)
            {
                dims[i] = d_type.extent(i);
            }
            hid_t res2 = res;

            H5_ERROR(res2 = H5Tarray_create(res, d_type.rank(), dims));

            if (H5Tcommitted(res) > 0) H5_ERROR(H5Tclose(res));

            res = res2;
        }
    }
    else
    {
        H5_ERROR(res = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte()));

        for (auto const &item : d_type.members())
        {
            hid_t t_member = convert_data_type_sp_to_h5(std::get<0>(item), true);

            H5_ERROR(H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item),
                               t_member));
            if (H5Tcommitted(t_member) > 0) H5_ERROR(H5Tclose(t_member));
        }
    }

    if (res == H5T_NO_CLASS)
    {
        WARNING << "sp.DataType convert to H5.DataType failed!" << std::endl;
        throw std::bad_cast();
    }
    return (res);
}

data_model::DataType convert_data_type_h5_to_sp(hid_t t_id)
{

    bool bad_cast_error = true;

    data_model::DataType dtype;

    H5T_class_t type_class = H5Tget_class(t_id);

    if (type_class == H5T_NO_CLASS) { bad_cast_error = true; }
    else if (type_class == H5T_OPAQUE)
    {
        data_model::DataType(std::type_index(typeid(void)), H5Tget_size(t_id)).swap(dtype);
    }
    else if (type_class == H5T_COMPOUND)
    {
        for (int i = 0, num = H5Tget_nmembers(t_id); i < num; ++i)
        {
            dtype.push_back(
                convert_data_type_h5_to_sp(H5Tget_member_type(t_id, i)),
                std::string(H5Tget_member_name(t_id, i)),
                static_cast<int>(  H5Tget_member_offset(t_id, i)));
        }

    }
    else if (type_class == H5T_INTEGER || type_class == H5T_FLOAT || type_class == H5T_ARRAY)
    {
        UNIMPLEMENTED;

//        hid_t atomic_id = H5Tget_native_type(type_class, H5T_DIR_ASCEND);
//		switch (atomic_id)
//		{
//		case H5T_NATIVE_CHAR:
//			make_datatype<char>().swap(dtype);
//			break;
//		case H5T_NATIVE_SHORT:
//			make_datatype<short>().swap(dtype);
//			break;
//		case H5T_NATIVE_INT:
//			make_datatype<int>().swap(dtype);
//			break;
//		case H5T_NATIVE_LONG:
//			make_datatype<long>().swap(dtype);
//			break;
//		case H5T_NATIVE_LLONG:
//			make_datatype<long long>().swap(dtype);
//			break;
//		case H5T_NATIVE_UCHAR:
//			make_datatype<unsigned char>().swap(dtype);
//			break;
//		case H5T_NATIVE_USHORT:
//			make_datatype<unsigned short>().swap(dtype);
//			break;
//		case H5T_NATIVE_UINT:
//			make_datatype<unsigned int>().swap(dtype);
//			break;
//		case H5T_NATIVE_ULONG:
//			make_datatype<unsigned long>().swap(dtype);
//			break;
//		case H5T_NATIVE_ULLONG:
//			make_datatype<unsigned long long>().swap(dtype);
//			break;
//		case H5T_NATIVE_FLOAT:
//			make_datatype<float>().swap(dtype);
//			break;
//		case H5T_NATIVE_DOUBLE:
//			make_datatype<double>().swap(dtype);
//			break;
//		case H5T_NATIVE_LDOUBLE:
//			make_datatype<long double>().swap(dtype);
//			break;
//		default:
//			bad_cast_error = true;
//
//		}
//        H5_ERROR(H5Tclose(atomic_id));
    }
    else if (type_class == H5T_TIME) { UNIMPLEMENTED; }
    else if (type_class == H5T_STRING) { UNIMPLEMENTED; }
    else if (type_class == H5T_BITFIELD) { UNIMPLEMENTED; }
    else if (type_class == H5T_REFERENCE) { UNIMPLEMENTED; }
    else if (type_class == H5T_ENUM) { UNIMPLEMENTED; }
    else if (type_class == H5T_VLEN) { UNIMPLEMENTED; }

    if (type_class == H5T_ARRAY)
    {
        int rank = H5Tget_array_ndims(t_id);
        hsize_t dims[rank];
        size_t dims2[rank];
        for (int i = 0; i < rank; ++i)
        {
            dims2[i] = dims[i];
        }
        H5_ERROR(H5Tget_array_dims(t_id, dims));

        dtype.extent(rank, dims2);
    }
    if (bad_cast_error)
    {
        logger::Logger(logger::LOG_ERROR) << "H5 DataType convert to sp.DataType failed!"
                                          << std::endl;
        throw std::bad_cast();
    }

    return std::move(dtype);

}

hid_t convert_data_space_sp_to_h5(data_model::DataSpace const &ds, size_t flag)
{

    int ndims = 0;

    typedef nTuple<hsize_t, MAX_NDIMS_OF_ARRAY> index_tuple;

    index_tuple dims;

    index_tuple start;

    index_tuple stride;

    index_tuple count;

    index_tuple block;

    std::tie(ndims, dims, start, stride, count, block) = ds.shape();


    if ((flag & SP_RECORD) != 0UL)
    {
        dims[ndims] = 1;
        start[ndims] = 0;
        count[ndims] = 1;
        stride[ndims] = 1;
        block[ndims] = 1;
        ++ndims;
    }

    index_tuple max_dims;

    max_dims = dims;

    if ((flag & SP_APPEND) != 0UL)
    {
        max_dims[ndims - 1] = H5S_UNLIMITED;
    }
    else if ((flag & SP_RECORD) != 0UL)
    {
        max_dims[ndims - 1] = H5S_UNLIMITED;
    }
    hid_t res = H5Screate_simple(ndims, &dims[0], &max_dims[0]);

    if (ds.is_simple())
    {
        H5_ERROR(H5Sselect_hyperslab(res, H5S_SELECT_SET, &start[0], &stride[0], &count[0], &block[0]));
    }
    else
    {
        size_t num_elements = ds.selected_points().size();

        std::vector<hsize_t> coords;

        int r_ndims = std::get<0>(ds.shape());

        auto const &idx = ds.selected_points();

        for (int i = 0; i < num_elements; ++i)
        {
            for (size_t j = 0; j < r_ndims; ++j)
            {
                coords.push_back(static_cast<hsize_t>(idx[i * r_ndims + j]));
            }
            if ((flag & SP_RECORD) != 0UL) { coords.push_back(0); }
        }
        H5Sselect_elements(res, H5S_SELECT_SET, coords.size() / ndims, &coords[0]);
    }
    return res;

}

data_model::DataSpace convert_data_space_h5_to_sp(hid_t)
{
    UNIMPLEMENTED;

    return data_model::DataSpace();
}

void HDF5Stream::push_buffer(std::string const &url, data_model::DataSet const &ds)
{

    std::string full_path = absolute_path(url);

    auto &item = m_pimpl_->m_buffer_map_[full_path];

    if (item.memory_space.is_full() && item.data != nullptr)
    {
        write(full_path, item, SP_APPEND);


        auto &d_shape = item.data_space.shape();
        int &d_ndims = std::get<0>(d_shape);
        std::get<1>(d_shape)[d_ndims - 1] = 0; //topology_dims
        std::get<4>(d_shape)[d_ndims - 1] = 0; //count


        auto &m_shape = item.memory_space.shape();
        int m_ndims = std::get<0>(m_shape);
        std::get<1>(m_shape)[m_ndims - 1] = 0; //topology_dims
        std::get<4>(m_shape)[m_ndims - 1] = 0; //count


    }
    else if (item.data == nullptr)
    {
        if (ds.memory_space.is_simple())
        {
            int ndims = 0;

            item.data_space = ds.data_space;
            item.memory_space = ds.memory_space;

            auto &d_shape = item.data_space.shape();

            int &d_ndims = std::get<0>(d_shape);

            ++d_ndims;

            std::get<1>(d_shape)[d_ndims - 1] = 0; //topology_dims
            std::get<2>(d_shape)[d_ndims - 1] = 0; //start
            std::get<3>(d_shape)[d_ndims - 1] = 1; //stride
            std::get<4>(d_shape)[d_ndims - 1] = 0; //count
            std::get<5>(d_shape)[d_ndims - 1] = 1; //block


            auto &m_shape = item.memory_space.shape();
            int &m_ndims = std::get<0>(m_shape);
            std::get<1>(m_shape) = std::get<4>(m_shape);
            ++m_ndims;

            std::get<1>(m_shape)[m_ndims - 1] = pimpl_s::DEFAULT_MAX_BUFFER_DEPTH; //topology_dims
            std::get<2>(m_shape)[m_ndims - 1] = 0; //start
            std::get<3>(m_shape)[m_ndims - 1] = 1; //stride
            std::get<4>(m_shape)[m_ndims - 1] = 0; //count
            std::get<5>(m_shape)[m_ndims - 1] = 1; //block

        }
        else
        {
            UNIMPLEMENTED;
        }


        item.data_type = ds.data_type;

        item.data = sp_alloc_memory(item.memory_space.size() * item.data_type.size_in_byte());

    }


    if (item.data != nullptr)
    {
        auto &d_shape = item.data_space.shape();
        int d_ndims = std::get<0>(d_shape);

        auto &m_shape = item.memory_space.shape();
        int m_ndims = std::get<0>(m_shape);

        size_t dtype_size = item.data_type.number_of_entities();


        if (ds.memory_space.is_simple())
        {

            auto const &src_shape = ds.memory_space.shape();

            size_t num_element = ds.memory_space.num_of_elements();

            int src_ndims = std::get<0>(src_shape);
            auto src_dims = std::get<1>(src_shape);
            auto src_start = std::get<2>(src_shape);
            auto src_count = std::get<4>(src_shape);

            int dest_ndims = std::get<0>(m_shape);
            auto dest_dims = std::get<1>(m_shape);
            auto dest_start = std::get<2>(m_shape);
            auto dest_count = std::get<4>(m_shape);


            // copy
            char *dest_p = reinterpret_cast<char *>(item.data.get());
            char const *src_p = reinterpret_cast<char *>(ds.data.get());


            dest_start[dest_ndims - 1] = ++dest_count[dest_ndims - 1];

            int ndims = src_ndims;
            auto dims = src_count;
            auto idx = dims;
            idx = 0;


            while (1)
            {

                size_t dest_s = 0;
                size_t src_s = 0;
                size_t dest_stride = 1;
                size_t src_stride = 1;
                for (int i = 0, ie = ndims; i < ndims; ++i)
                {
                    dest_s = dest_s * dest_stride + (dest_start[i] + idx[i]);
                    src_s = dest_s * src_stride + (dest_start[i] + idx[i]);

                    dest_stride *= dest_dims[i];
                    src_stride *= src_dims[i];
                }

                dest_s *= dtype_size;
                src_s *= dtype_size;

                for (size_t i = 0, ie = dtype_size; i < ie; ++i) { dest_p[dest_s + i] = src_p[src_s + i]; }


                ++idx[0];

                int n = 0;
                while (n < ndims)
                {
                    ++idx[n];

                    if (idx[n] < dims[n])
                    {
                        break;
                    }
                    else
                    {
                        idx[n] = 0;
                        ++n;
                    }
                }

                if (n >= ndims) { break; }
            }

        }
        else
        {

            UNIMPLEMENTED;
        }


        ++std::get<1>(d_shape)[d_ndims - 1]; //count
        ++std::get<4>(d_shape)[d_ndims - 1]; //count
        ++std::get<4>(m_shape)[m_ndims - 1]; //count
    }

}

std::string HDF5Stream::write_buffer(std::string const &url, bool is_forced_flush)
{
    std::string full_path = absolute_path(url);

    std::string res = "";

    auto it = m_pimpl_->m_buffer_map_.find(full_path);

    if (it == m_pimpl_->m_buffer_map_.end()) { return res; }


    int ndims = std::get<0>(it->second.memory_space.shape());
    auto count = std::get<4>(it->second.memory_space.shape());

    if ((is_forced_flush && count[ndims - 1] > 0) || count[ndims - 1] == pimpl_s::DEFAULT_MAX_BUFFER_DEPTH)
    {
        res = write(full_path, it->second, SP_APPEND);

        std::get<1>(it->second.data_space.shape())[ndims - 1] = 0;
        std::get<4>(it->second.data_space.shape())[ndims - 1] = 0;
        std::get<4>(it->second.memory_space.shape())[ndims - 1] = 0;

        VERBOSE << "======= Flush Buffer to : " << res << std::endl;
    }
    else
    {
//        VERBOSE << "Push m_data to m_buffer : " << full_path << std::endl;
    }

    return res;

}

void HDF5Stream::flush()
{

    for (auto const &item:m_pimpl_->m_buffer_map_)
    {
        write_buffer(item.first, true);
    }

}

std::string HDF5Stream::write(std::string const &url, data_model::DataSet const &ds, int flag)
{


    if ((flag & SP_BUFFER) != 0)
    {
        push_buffer(url, ds);
        return write_buffer(url);
    }

    if ((ds.data == nullptr) || ds.memory_space.size() == 0)
    {
        VERBOSE << "ignore empty m_data set! " << url << std::endl;
        return "";
    }
    typedef nTuple<hsize_t, MAX_NDIMS_OF_ARRAY> index_tuple;

    if (!ds.is_valid())
    {
        WARNING << "Invalid Data Set! "
                << "[ URL = \"" << url << "\"," << std::endl
                << " Data is " << ((ds.data != nullptr) ? "valid" : "invalid ") << ". " << std::endl
                << " DataType is " << ((ds.data_type.is_valid()) ? "valid" : "invalid") << ". " << std::endl
                << " File Space is " << ((ds.data_space.is_valid()) ? "valid" : "invalid") << ". size="
                << ds.data_space.num_of_elements() << ". " << std::endl
                << " Memory Space is " << ((ds.memory_space.is_valid()) ? "valid" : "invalid") << ".  size="
                << ds.memory_space.num_of_elements() << ". " << std::endl
                << " ]"

                << std::endl;
        return "";
    }


    std::string dsname = "";

    bool is_existed = false;

    std::tie(is_existed, dsname) = this->open(url, flag);

    hid_t d_type = convert_data_type_sp_to_h5(ds.data_type);

    hid_t m_space = convert_data_space_sp_to_h5(ds.memory_space, SP_NEW);

    hid_t f_space = convert_data_space_sp_to_h5(ds.data_space, flag);

    hid_t dset;

    if (!is_existed)
    {

        hid_t dcpl_id = H5P_DEFAULT;

        if ((flag & (SP_APPEND | SP_RECORD)) != 0)
        {
            index_tuple current_dims;

            int f_ndims = H5Sget_simple_extent_ndims(f_space);

            H5_ERROR(H5Sget_simple_extent_dims(f_space, &current_dims[0], nullptr));

            H5_ERROR(dcpl_id = H5Pcreate(H5P_DATASET_CREATE));

            H5_ERROR(H5Pset_chunk(dcpl_id, f_ndims, &current_dims[0]));
        }

        H5_ERROR(dset = H5Dcreate(m_pimpl_->base_group_id_, dsname.c_str(), //
                                  d_type, f_space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT));

        if (dcpl_id != H5P_DEFAULT) { H5_ERROR(H5Pclose(dcpl_id)); }

        H5_ERROR(H5Fflush(m_pimpl_->base_group_id_, H5F_SCOPE_GLOBAL));
    }
    else
    {

        H5_ERROR(dset = H5Dopen(m_pimpl_->base_group_id_, dsname.c_str(), H5P_DEFAULT));

        index_tuple current_dimensions;

        hid_t current_f_space;

        H5_ERROR(current_f_space = H5Dget_space(dset));

        int current_ndims = H5Sget_simple_extent_dims(current_f_space,
                                                      &current_dimensions[0], nullptr);

        H5_ERROR(H5Sclose(current_f_space));

        index_tuple new_f_dimensions;
        index_tuple new_f_max_dimensions;
        index_tuple new_f_offset;
        index_tuple new_f_end;

        int new_f_ndims = H5Sget_simple_extent_dims(f_space,
                                                    &new_f_dimensions[0], &new_f_max_dimensions[0]);

        H5_ERROR(H5Sget_select_bounds(f_space, &new_f_offset[0], &new_f_end[0]));

        nTuple<hssize_t, MAX_NDIMS_OF_ARRAY> new_f_offset2;

        new_f_offset2 = 0;

        if ((flag & SP_APPEND) != 0)
        {

            new_f_dimensions[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

            new_f_offset2 = 0;

            new_f_offset2[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

        }
        else if ((flag & SP_RECORD) != 0)
        {
            new_f_dimensions[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

            new_f_offset2 = 0;

            new_f_offset2[new_f_ndims - 1] = current_dimensions[new_f_ndims - 1];

        }

        H5_ERROR(H5Dset_extent(dset, &new_f_dimensions[0]));

        H5_ERROR(H5Sset_extent_simple(f_space, new_f_ndims, &new_f_dimensions[0], &new_f_max_dimensions[0]));

        H5_ERROR(H5Soffset_simple(f_space, &new_f_offset2[0]));

    }

// create property list for collective dataset write.
    if (GLOBAL_COMM.is_valid())
    {
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5_ERROR(H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT));
        H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, plist_id, ds.data.get()));
        H5_ERROR(H5Pclose(plist_id));
    }
    else
    {
        H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, ds.data.get()));
    }

    m_pimpl_->set_attribute(dset, ds.properties);

    H5_ERROR(H5Dclose(dset));

    if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));

    if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));

    if (H5Tcommitted(d_type) > 0) { H5_ERROR(H5Tclose(d_type)); }

    VERBOSE << "Write DataSet to hdf5 file:" << pwd() + dsname << std::endl;

    return pwd() + dsname;
}

std::string HDF5Stream::read(std::string const &url, data_model::DataSet *ds, int flag)
{
    UNIMPLEMENTED;
    return "UNIMPLEMENTED";
}
//hid_t HDF5Stream::pimpl_s::create_h5_dataset(dataset const & ds,
//		size_t id) const
//{
//
//	h5_dataset res;
//
//	res.m_data = ds.m_data;
//
//	res.DataType = ds.DataType;
//
//	res.id = id;
//
//	res.m_ndims_ = ds.data_space.topology_num_of_dims();
//
//	std::tie(res.f_start, res.f_count) = ds.data_space.global_shape();
//
//	std::tie(res.m_start, res.m_count) = ds.data_space.local_shape();
//
//	std::tie(res.start, res.count, res.stride, res.block) =
//			ds.data_space.m_global_dims_();
//
//	if ((id & SP_UNORDER) == SP_UNORDER)
//	{
//		std::tie(res.f_start[0], res.f_count[0]) = sync_global_location(
//				res.f_count[0]);
//
//		res.f_stride[0] = res.f_count[0];
//	}
//
//	if (ds.DataType.m_ndims_ > 0)
//	{
//		for (int j = 0; j < ds.DataType.m_ndims_; ++j)
//		{
//
//			res.f_count[res.m_ndims_ + j] = ds.DataType.dimensions_[j];
//			res.f_start[res.m_ndims_ + j] = 0;
//			res.f_stride[res.m_ndims_ + j] = res.f_count[res.m_ndims_ + j];
//
//			res.m_count[res.m_ndims_ + j] = ds.DataType.dimensions_[j];
//			res.m_start[res.m_ndims_ + j] = 0;
//			res.strides[res.m_ndims_ + j] = res.m_count[res.m_ndims_ + j];
//
//			res.count[res.m_ndims_ + j] = 1;
//			res.block[res.m_ndims_ + j] = ds.DataType.dimensions_[j];
//
//		}
//
//		res.m_ndims_ += ds.DataType.m_ndims_;
//	}
//
//	if (properties["Enable Compact Storage"].as<bool>(false))
//	{
//		res.id |= SP_APPEND;
//	}
//
//	if (properties["Force Record Storage"].as<bool>(false))
//	{
//		res.id |= SP_RECORD;
//	}
//	if (properties["Force Write CellCache"].as<bool>(false))
//	{
//		res.id |= SP_CACHE;
//	}
//	return std::move(res);
//
//}

//std::string HDF5Stream::pimpl_s::write(std::string const &url, h5_dataset ds)
//{
//	if ((ds.id & (SP_UNORDER)) == (SP_UNORDER))
//	{
//		return write_array(url, ds);
//	}
//
//	if ((ds.id & SP_RECORD) == SP_RECORD)
//	{
//		convert_record_dataset(&ds);
//	}
//
//	if ((ds.id & SP_CACHE) == SP_CACHE)
//	{
//		return write_cache(url, ds);
//	}
//	else
//	{
//		return write_array(url, ds);
//	}
//
//}

//void HDF5Stream::pimpl_s::convert_record_dataset(h5_dataset *pds) const
//{
//	for (int i = pds->m_ndims_; i > 0; --i)
//	{
//
//		pds->f_count[i] = pds->f_count[i - 1];
//		pds->f_start[i] = pds->f_start[i - 1];
//		pds->f_stride[i] = pds->f_stride[i - 1];
//		pds->m_count[i] = pds->m_count[i - 1];
//		pds->m_start[i] = pds->m_start[i - 1];
//		pds->strides[i] = pds->strides[i - 1];
//		pds->count[i] = pds->count[i - 1];
//		pds->block[i] = pds->block[i - 1];
//
//	}
//
//	pds->f_count[0] = 1;
//	pds->f_start[0] = 0;
//	pds->f_stride[0] = 1;
//
//	pds->m_count[0] = 1;
//	pds->m_start[0] = 0;
//	pds->strides[0] = 1;
//
//	pds->count[0] = 1;
//	pds->block[0] = 1;
//
//	++pds->m_ndims_;
//
//	pds->id |= SP_APPEND;
//
//}

//std::string HDF5Stream::pimpl_s::write_cache(std::string const & p_url,
//		h5_dataset const & ds)
//{
//
//	std::string filename, grp_name, dsname;
//
//	std::tie(filename, grp_name, dsname, std::ignore) = parser_url(p_url);
//
//	cd(filename, grp_name, ds.id);
//
//	std::string url = pwd() + dsname;
//
//	if (cache_.find(url) == cache_.end())
//	{
//		size_t cache_memory_size = ds.DataType.ele_size_in_byte_;
//		for (int i = 0; i < ds.m_ndims_; ++i)
//		{
//			cache_memory_size *= ds.m_count[i];
//		}
//
//		size_t cache_depth = properties["Max CellCache Size"].as<size_t>(
//				10 * 1024 * 1024UL) / cache_memory_size;
//
//		if (cache_depth <= properties["Min CellCache Number"].as<int>(5))
//		{
//			return write_array(url, ds);
//		}
//		else
//		{
//			sp_make_shared_array<byte_type>(cache_memory_size * cache_depth).swap(
//					std::get<0>(cache_[url]));
//
//			h5_dataset & item = std::get<1>(cache_[url]);
//
//			item = ds;
//
//			item.id |= SP_APPEND;
//
//			item.m_ndims_ = ds.m_ndims_;
//
//			item.count[0] = 0;
//			item.m_count[0] = item.strides[0] * cache_depth + item.m_start[0];
//			item.f_count[0] = item.f_stride[0] * cache_depth + item.f_start[0];
//
//		}
//	}
//	auto & m_data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	size_t memory_size = ds.DataType.ele_size_in_byte_ * item.strides[0];
//
//	for (int i = 1; i < item.m_ndims_; ++i)
//	{
//		memory_size *= item.m_count[i];
//	}
//
//	std::memcpy(
//			reinterpret_cast<void*>(m_data.get() + item.count[0] * memory_size),
//			ds.m_data.get(), memory_size);
//
//	++item.count[0];
//
//	if (item.count[0] * item.f_stride[0] + item.f_start[0] >= item.m_count[0])
//	{
//		return flush_cache(url);
//	}
//	else
//	{
//		return "\"" + url + "\" is write to cache";
//	}
//
//}
//std::string HDF5Stream::pimpl_s::flush_cache(std::string const & url)
//{
//
//	if (cache_.find(url) == cache_.end())
//	{
//		return url + " is not found !";
//	}
//
//	auto & m_data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	hsize_t t_f_shape = item.f_count[0];
//	hsize_t t_m_shape = item.m_count[0];
//
//	item.m_count[0] = item.count[0] * item.strides[0] + item.m_start[0];
//	item.f_count[0] = item.count[0] * item.f_stride[0] + item.f_start[0];
//
//	auto res = write_array(url, item);
//
//	item.m_count[0] = t_f_shape;
//	item.f_count[0] = t_m_shape;
//
//	item.count[0] = 0;
//
//	return res;
//}

//=====================================================================================

}}// namespace simpla

