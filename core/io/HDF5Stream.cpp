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

#include <cstring> //for memcopy

#include "HDF5Stream.h"
#include "../data_model/DataSet.h"


#include "../parallel/Parallel.h"
#include "../parallel/MPIComm.h"
#include "../parallel/MPIAuxFunctions.h"


#include "../gtl/utilities/utilities.h"
#include "../gtl/utilities/memory_pool.h"

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


    hid_t base_file_id_;
    hid_t base_group_id_;


    void set_attribute(hid_t loc_id, std::string const &name, any const &v);

    void set_attribute(hid_t loc_id, Properties const &v);

    Properties get_attribute(hid_t loc_id, std::string const &name) const;

    Properties get_attribute(hid_t loc_id, int idx = -1) const;

    static constexpr size_t DEFAULT_MAX_BUFFER_DEPTH = 1024;

    std::map<std::string, data_model::DataSet> m_buffer_map_;


};

HDF5Stream::HDF5Stream() : m_pimpl_(new pimpl_s) { }

HDF5Stream::~HDF5Stream()
{
    for (auto const &item:m_pimpl_->m_buffer_map_)
    {
        write_buffer(item.first, true);
    }
}

bool HDF5Stream::is_valid() const
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

std::tuple<bool, std::string> HDF5Stream::open(std::string const &url, size_t flag)
{
    std::string file_name = IOStream::current_file_name();
    std::string grp_name = IOStream::current_group_name();
    std::string obj_name = "";

    if (url != "")
    {
        std::tie(file_name, grp_name, obj_name, std::ignore) = IOStream::parser_url(url);
    }

    //TODO using regex parser url


    if (IOStream::current_file_name() != file_name)
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

    data_model::DataType dtype = any_v.data_type();

    void const *v = any_v.data();

    std::string file_name, grp_path, obj_name, attr_name;

    std::tie(file_name, grp_path, obj_name, attr_name) = IOStream::parser_url(url);

    hid_t g_id, o_id;
    UNIMPLEMENTED;
// FIXME
//    std::tie(grp_path, g_id) = m_pimpl_->open_group(grp_path);
//
//    if (o_id != g_id) { H5Oclose(o_id); }
//
//    if (g_id != m_pimpl_->base_group_id_) { H5Gclose(g_id); }

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
        hid_t m_type = convert_data_type_sp_to_h5(any_v.data_type());

        hid_t m_space = H5Screate(H5S_SCALAR);

        hid_t a_id = H5Acreate(loc_id, name.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);

        H5Awrite(a_id, m_type, any_v.data());

        if (H5Tcommitted(m_type) > 0)
        {
            H5Tclose(m_type);
        }

        H5Aclose(a_id);

        H5Sclose(m_space);
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

    if (obj_name != "")
    {

        // FIXME
//        hid_t g_id;
//        std::tie(grp_name, g_id) = m_pimpl_->open_group(grp_name);
//
//        if (H5Aexists_by_name(g_id, obj_name.c_str(), attr_name.c_str(), H5P_DEFAULT))
//        {
//            H5Adelete_by_name(g_id, obj_name.c_str(), attr_name.c_str(), H5P_DEFAULT);
//        }
//        if (g_id != m_pimpl_->base_group_id_)
//        {
//            H5Gclose(g_id);
//        }
    }
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

        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id));

        H5_ERROR(H5Pclose(plist_id));

    }
    else
    {
        H5_ERROR(f_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
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

    if (!d_type.is_valid()) THROW_EXCEPTION_RUNTIME_ERROR("illegal data type");

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
            RUNTIME_ERROR << "Unknown data type:" << d_type.name();
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
        max_dims[0] = H5S_UNLIMITED;
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
    typedef nTuple<data_model::DataSpace::index_type, MAX_NDIMS_OF_ARRAY> index_tuple;

    std::string full_path = absolute_path(url);

    auto &item = m_pimpl_->m_buffer_map_[full_path];

    if (item.data_space.is_full() && item.data != nullptr)
    {
        write(full_path, item, SP_APPEND);
        item.data_space.clear_selected();
    }
    else if (item.data == nullptr)
    {
        if (ds.data_space.is_simple())
        {
            int ndims = 0;

            index_tuple count;

            std::tie(ndims, std::ignore /* dims*/, std::ignore /* start*/, std::ignore /*stride*/, count,
                     std::ignore /*block*/) = ds.data_space.shape();

            count[ndims] = pimpl_s::DEFAULT_MAX_BUFFER_DEPTH;

            item.data_space = data_model::DataSpace::create_simple(ndims + 1, &count[0]);

        }
        else
        {
            data_model::DataSpace::index_type dims[2] = {ds.data_space.size(), pimpl_s::DEFAULT_MAX_BUFFER_DEPTH};

            item.data_space = data_model::DataSpace::create_simple(2, dims);
        }

        index_tuple count;

        count = 0;

        item.data_space.select_hyperslab(&count[0], nullptr, &count[0], nullptr);

        item.data_type = ds.data_type;
        CHECK(item.data_space.size() * item.data_type.size_in_byte());
        item.data = sp_alloc_memory(item.data_space.size() * item.data_type.size_in_byte());

    }
    ASSERT(item.data != nullptr);


    int dest_ndims = 0;
    index_tuple dest_dims;
    index_tuple dest_start;
    index_tuple dest_count;

    std::tie(dest_ndims, dest_dims, dest_start, std::ignore /*stride*/, dest_count,
             std::ignore /*block*/ ) = item.data_space.shape();


    if (ds.data_space.is_simple())
    {
        int src_ndims = 0;
        index_tuple src_dims;
        index_tuple src_start;
        index_tuple src_count;

        std::tie(src_ndims, src_dims, src_start, std::ignore /*stride*/, src_count,
                 std::ignore /*block*/ ) = ds.data_space.shape();

        // copy

        if (!ds.data_space.is_full())
        {
            // FIXME (!ds.data_space.is_full())
            UNIMPLEMENTED;
        }
        else
        {

            char *dest_p = reinterpret_cast<char *>(item.data.get()) +
                           item.data_space.num_of_elements() * item.data_type.size();

            char *src_p = reinterpret_cast<char *>(ds.data.get());

            CHECK(ds.data_space.num_of_elements() * ds.data_type.size());

//            std::strncpy(dest_p, src_p, ds.data_space.num_of_elements() * ds.data_type.size());

            for (size_t i = 0, ie = ds.data_space.num_of_elements() * ds.data_type.size(); i < ie; ++i)
            {
                dest_p[i] = src_p[i];
            }
        }
    }
    else
    {
        data_model::DataSpace::index_type dims[2] = {ds.data_space.size(), pimpl_s::DEFAULT_MAX_BUFFER_DEPTH};
        auto const &selected_points = ds.data_space.selected_points();
        const size_t type_size = ds.data_type.size();
        char *dest_p = reinterpret_cast<char *>(item.data.get()) +
                       item.data_space.num_of_elements() * item.data_type.size();

        char *src_p = reinterpret_cast<char *>(ds.data.get());
        for (size_t n = 0, ne = selected_points.size(); n < ne; ++n)
        {
            for (size_t i = 0, ie = type_size; i < ie; ++i)
            {
                dest_p[n * type_size + i] = src_p[selected_points[n] * type_size + i];
            }

        }
    }
    ++dest_count[dest_ndims - 1];
    item.data_space.select_hyperslab(&dest_start[0], nullptr, &dest_count[0], nullptr);


}

std::string HDF5Stream::write_buffer(std::string const &url, bool is_forced_flush)
{
    std::string full_path = absolute_path(url);

    std::string res = "";

    auto it = m_pimpl_->m_buffer_map_.find(full_path);

    if (it != m_pimpl_->m_buffer_map_.end() && (it->second.data_space.is_full() || is_forced_flush))
    {
        res = write(full_path, it->second, SP_APPEND);

        it->second.data_space.clear_selected();
    }
    return res;


}

std::string HDF5Stream::write(std::string const &url, data_model::DataSet const &ds, size_t flag)
{


    if ((flag & SP_BUFFER) != 0)
    {
        push_buffer(url, ds);
        return write_buffer(url);
    }

    if ((ds.data == nullptr) || ds.memory_space.size() == 0)
    {
        VERBOSE << "ignore empty data set" << std::endl;
        return "";
    }
    typedef nTuple<hsize_t, MAX_NDIMS_OF_ARRAY> index_tuple;

    if (!ds.is_valid())
    {
        WARNING << "Invalid dataset! "
        << "[ URL = \"" << url << "\","
        << " Data is " << ((ds.data != nullptr) ? "not" : " ") << " empty. "
        << " Datatype is " << ((ds.data_type.is_valid()) ? "" : "not") << " valid. "
        << " Data Space is " << ((ds.data_space.is_valid()) ? "" : "not")
        << " valid. size=" << ds.data_space.num_of_elements()
        << " Memory Space is " << ((ds.memory_space.is_valid()) ? "" : "not") << " valid.  size=" <<
        ds.memory_space.num_of_elements()
        << " Space is " << ((ds.memory_space.is_valid()) ? "" : "not") << " valid."
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

            new_f_dimensions[0] += current_dimensions[0];

            new_f_offset2 = 0;

            new_f_offset2[0] += current_dimensions[0];

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

// create property list for collective data_set write.
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

std::string HDF5Stream::read(std::string const &url, data_model::DataSet *ds, size_t flag)
{
    UNIMPLEMENTED;
    return "UNIMPLEMENTED";
}
//hid_t HDF5Stream::pimpl_s::create_h5_dataset(data_set const & ds,
//		size_t flag) const
//{
//
//	h5_dataset res;
//
//	res.data = ds.data;
//
//	res.DataType = ds.DataType;
//
//	res.flag = flag;
//
//	res.ndims = ds.data_space.num_of_dims();
//
//	std::tie(res.f_start, res.f_count) = ds.data_space.global_shape();
//
//	std::tie(res.m_start, res.m_count) = ds.data_space.local_shape();
//
//	std::tie(res.start, res.count, res.stride, res.block) =
//			ds.data_space.shape();
//
//	if ((flag & SP_UNORDER) == SP_UNORDER)
//	{
//		std::tie(res.f_start[0], res.f_count[0]) = sync_global_location(
//				res.f_count[0]);
//
//		res.f_stride[0] = res.f_count[0];
//	}
//
//	if (ds.DataType.ndims > 0)
//	{
//		for (int j = 0; j < ds.DataType.ndims; ++j)
//		{
//
//			res.f_count[res.ndims + j] = ds.DataType.dimensions_[j];
//			res.f_start[res.ndims + j] = 0;
//			res.f_stride[res.ndims + j] = res.f_count[res.ndims + j];
//
//			res.m_count[res.ndims + j] = ds.DataType.dimensions_[j];
//			res.m_start[res.ndims + j] = 0;
//			res.m_stride[res.ndims + j] = res.m_count[res.ndims + j];
//
//			res.count[res.ndims + j] = 1;
//			res.block[res.ndims + j] = ds.DataType.dimensions_[j];
//
//		}
//
//		res.ndims += ds.DataType.ndims;
//	}
//
//	if (properties["Enable Compact Storage"].as<bool>(false))
//	{
//		res.flag |= SP_APPEND;
//	}
//
//	if (properties["Force Record Storage"].as<bool>(false))
//	{
//		res.flag |= SP_RECORD;
//	}
//	if (properties["Force Write Cache"].as<bool>(false))
//	{
//		res.flag |= SP_CACHE;
//	}
//	return std::move(res);
//
//}

//std::string HDF5Stream::pimpl_s::write(std::string const &url, h5_dataset ds)
//{
//	if ((ds.flag & (SP_UNORDER)) == (SP_UNORDER))
//	{
//		return write_array(url, ds);
//	}
//
//	if ((ds.flag & SP_RECORD) == SP_RECORD)
//	{
//		convert_record_dataset(&ds);
//	}
//
//	if ((ds.flag & SP_CACHE) == SP_CACHE)
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
//	for (int i = pds->ndims; i > 0; --i)
//	{
//
//		pds->f_count[i] = pds->f_count[i - 1];
//		pds->f_start[i] = pds->f_start[i - 1];
//		pds->f_stride[i] = pds->f_stride[i - 1];
//		pds->m_count[i] = pds->m_count[i - 1];
//		pds->m_start[i] = pds->m_start[i - 1];
//		pds->m_stride[i] = pds->m_stride[i - 1];
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
//	pds->m_stride[0] = 1;
//
//	pds->count[0] = 1;
//	pds->block[0] = 1;
//
//	++pds->ndims;
//
//	pds->flag |= SP_APPEND;
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
//	cd(filename, grp_name, ds.flag);
//
//	std::string url = pwd() + dsname;
//
//	if (cache_.find(url) == cache_.end())
//	{
//		size_t cache_memory_size = ds.DataType.ele_size_in_byte_;
//		for (int i = 0; i < ds.ndims; ++i)
//		{
//			cache_memory_size *= ds.m_count[i];
//		}
//
//		size_t cache_depth = properties["Max Cache Size"].as<size_t>(
//				10 * 1024 * 1024UL) / cache_memory_size;
//
//		if (cache_depth <= properties["Min Cache Number"].as<int>(5))
//		{
//			return write_array(url, ds);
//		}
//		else
//		{
//			sp_make_shared_array<ByteType>(cache_memory_size * cache_depth).swap(
//					std::get<0>(cache_[url]));
//
//			h5_dataset & item = std::get<1>(cache_[url]);
//
//			item = ds;
//
//			item.flag |= SP_APPEND;
//
//			item.ndims = ds.ndims;
//
//			item.count[0] = 0;
//			item.m_count[0] = item.m_stride[0] * cache_depth + item.m_start[0];
//			item.f_count[0] = item.f_stride[0] * cache_depth + item.f_start[0];
//
//		}
//	}
//	auto & data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	size_t memory_size = ds.DataType.ele_size_in_byte_ * item.m_stride[0];
//
//	for (int i = 1; i < item.ndims; ++i)
//	{
//		memory_size *= item.m_count[i];
//	}
//
//	std::memcpy(
//			reinterpret_cast<void*>(data.get() + item.count[0] * memory_size),
//			ds.data.get(), memory_size);
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
//	auto & data = std::get<0>(cache_[url]);
//	auto & item = std::get<1>(cache_[url]);
//
//	hsize_t t_f_shape = item.f_count[0];
//	hsize_t t_m_shape = item.m_count[0];
//
//	item.m_count[0] = item.count[0] * item.m_stride[0] + item.m_start[0];
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
