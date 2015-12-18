/**
 * @file mpi_datatype.cpp
 *
 * @date 2014-7-8
 * @author salmon
 */
#include <mpi.h>
#include <typeinfo>
#include <typeindex>
#include "mpi_comm.h"
#include "mpi_datatype.h"
#include "../gtl/utilities/utilities.h"
#include "../gtl/utilities/pretty_stream.h"

namespace simpla
{

MPIDataType::MPIDataType()
{
}

MPIDataType::MPIDataType(MPIDataType const &other)
{
    MPI_ERROR(MPI_Type_dup(other.type(), &m_type_));
}

void MPIDataType::swap(MPIDataType &other)
{
    std::swap(m_type_, other.m_type_);
    std::swap(is_commited_, other.is_commited_);
}

MPIDataType::~MPIDataType()
{
    if (is_commited_)
    {
        MPI_ERROR(MPI_Type_free(&m_type_));
    }
}

MPIDataType MPIDataType::create(data_model::DataType const &data_type, //
                                int ndims, //
                                size_t const *p_dims,        //
                                size_t const *p_start,      //
                                size_t const *p_stride,      //
                                size_t const *p_count,       //
                                size_t const *p_block,       //
                                bool c_order_array)
{


    MPI_Datatype res_type;

    bool is_predefined = true;


    if (data_type.is_compound())
    {
        is_predefined = false;
        //TODO create MPI structure DataType
        //		MPI_Type_contiguous(DataType.ele_size_in_byte(), MPI_BYTE,
        //				&res.m_type_);

        ////		int MPI_Type_create_struct(
        ////		  int count,
        ////		  int array_of_blocklengths[],
        ////		  MPI_Aint array_of_displacements[],
        ////		  MPI_Datatype array_of_types[],
        ////		  MPI_Datatype *newtype
        ////		);
        std::vector<MPIDataType> dtypes;
        std::vector<int> array_of_block_lengths;
        std::vector<MPI_Aint> array_of_displacements;
        std::vector<MPI_Datatype> array_of_types;
        //		  MPI_Aint array_of_displacements[],
        //		  MPI_Datatype array_of_types[],
        for (auto const &item : data_type.members())
        {
            data_model::DataType sub_datatype;

            int offset;

            std::tie(sub_datatype, std::ignore, offset) = item;

            int block_length = 1;

            for (int i = 0; i < sub_datatype.rank(); ++i)
            {
                block_length *= sub_datatype.extent(i);
            }

            dtypes.push_back(MPIDataType::create(sub_datatype.element_type()));

            array_of_block_lengths.push_back(block_length);
            array_of_displacements.push_back(offset);
            array_of_types.push_back(dtypes.rbegin()->type());

        }

        MPI_Type_create_struct(        //
                static_cast<int>(array_of_block_lengths.size()),        //
                &array_of_block_lengths[0],        //
                &array_of_displacements[0],        //
                &array_of_types[0], &res_type);

    }
    else if (data_type.template is_same<int>())
    {
        res_type = MPI_INT;
    }
    else if (data_type.template is_same<long>())
    {
        res_type = MPI_LONG;
    }
    else if (data_type.template is_same<unsigned int>())
    {
        res_type = MPI_UNSIGNED;
    }
    else if (data_type.template is_same<unsigned long>())
    {
        res_type = MPI_UNSIGNED_LONG;
    }
    else if (data_type.template is_same<float>())
    {
        res_type = MPI_FLOAT;
    }
    else if (data_type.template is_same<double>())
    {
        res_type = MPI_DOUBLE;
    }
    else if (data_type.template is_same<long double>())
    {
        res_type = MPI_LONG_DOUBLE;
    }
    else if (data_type.template is_same<std::complex<double>>())
    {
        res_type = MPI_2DOUBLE_COMPLEX;
    }
    else if (data_type.template is_same<std::complex<float>>())
    {
        res_type = MPI_2COMPLEX;
    }
    else
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Cannot create MPI DataType:" + data_type.name());
    }

    if (data_type.is_array() || (ndims > 0 && p_dims != nullptr))
    {

        int mdims = ndims + data_type.rank();

        nTuple<int, MAX_NDIMS_OF_ARRAY> l_dims;
        nTuple<int, MAX_NDIMS_OF_ARRAY> l_offset;
        nTuple<int, MAX_NDIMS_OF_ARRAY> l_stride;
        nTuple<int, MAX_NDIMS_OF_ARRAY> l_count;
        nTuple<int, MAX_NDIMS_OF_ARRAY> l_block;

        MPIDataType old_type = MPIDataType::create(data_type.element_type());

        if (p_dims != nullptr)
        {
            l_dims = p_dims;
        }

        if (p_start == nullptr)
        {
            l_offset = 0;
        }
        else
        {
            l_offset = p_start;
        }

        if (p_count == nullptr)
        {
            l_count = l_dims;
        }
        else
        {
            l_count = p_count;
        }

//        if (p_stride != nullptr || p_block != nullptr)
//        {
//            //TODO create mpi DataType with stride and block
//            UNIMPLEMENTED2("!! 'stride'  and 'block' are ignored! ");
//        }

        for (int i = 0; i < data_type.rank(); ++i)
        {
            l_dims[ndims + i] = static_cast<int>(data_type.extent(i));
            l_count[ndims + i] = static_cast<int>( data_type.extent(i));
            l_offset[ndims + i] = 0;
        }

        MPI_Datatype ele_type = res_type;


        MPI_Type_create_subarray(ndims + data_type.rank(),
                                 &l_dims[0], &l_count[0], &l_offset[0],
                                 (c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN),
                                 ele_type, &res_type);

        if (!is_predefined)
        {
            MPI_Type_free(&ele_type);
        }
        is_predefined = false;
    }

    if (!is_predefined)
    {
        MPI_ERROR(MPI_Type_commit(&res_type));
    }

    MPIDataType res;
    res.m_type_ = res_type;
    res.is_commited_ = !is_predefined;

    return std::move(res);
}

MPIDataType MPIDataType::create(data_model::DataType const &data_type, data_model::DataSpace const &d_space,
                                bool c_order_array)
{

    int ndims;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> dimensions, start, stride, count, block;

    std::tie(ndims, dimensions, start, stride, count, block) = d_space.shape();


    return create(data_type,
                  ndims,
                  &dimensions[0],
                  &start[0],
                  &stride[0],
                  &count[0],
                  &block[0],
                  c_order_array
    );
}

size_t MPIDataType::size() const
{
    int s = 0;
    MPI_ERROR(MPI_Type_size(m_type_, &s));
    return static_cast<size_t>(s);
}
}  // namespace simpla

