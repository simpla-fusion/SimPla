/**
 * @file dataset.cpp
 *
 *  Created on: 2014-12-12
 *      Author: salmon
 */
#include <ostream>
#include <string.h>

#include "dataset.h"
#include "../gtl/utilities/pretty_stream.h"
#include "../gtl/utilities/memory_pool.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "../parallel/mpi_update.h"
#include "../parallel/mpi_comm.h"

namespace simpla
{

void DataSet::deploy()
{
    if (!empty())
    {
        return;
    }
    else
    {
        data = SingletonHolder<MemoryPool>::instance().raw_alloc(
                datatype.size_in_byte() * dataspace.size());
    }
}

void DataSet::clear()
{
    deploy();
    memset(data.get(), 0, memory_space.size() * datatype.size_in_byte());
}


void DataSet::copy(void const *other)
{
    deploy();
    memcpy(data.get(), other, memory_space.size() * datatype.size_in_byte());
}

bool DataSet::is_same(void const *other) const
{
    return data.get() == other;
}

bool DataSet::is_equal(void const *other) const
{
    return is_same(other) ||
           memcmp(data.get(), other, memory_space.size() * datatype.size_in_byte()) != 0;
}

std::ostream &DataSet::print(std::ostream &os) const
{

    int ndims = 0;

    nTuple<size_t, 3> dims;

    std::tie(ndims, dims, std::ignore, std::ignore, std::ignore, std::ignore) = memory_space.shape();

    long d[ndims];

    for (int i = 0; i < ndims; ++i)
    {
        if (dims[i] > 1)
        {
            d[ndims] = dims[i];
            ++ndims;
        }

    }

    if (datatype.template is_same<int>())
    {
        printNdArray(os, reinterpret_cast<int *>(data.get()), ndims, d);
    }
    else if (datatype.template is_same<long>())
    {
        printNdArray(os, reinterpret_cast<long *>(data.get()), ndims, d);
    }
    else if (datatype.template is_same<unsigned long>())
    {
        printNdArray(os, reinterpret_cast<unsigned long *>(data.get()), ndims,
                     d);
    }
    else if (datatype.template is_same<float>())
    {
        printNdArray(os, reinterpret_cast<float *>(data.get()), ndims, d);
    }
    else if (datatype.template is_same<double>())
    {
        printNdArray(os, reinterpret_cast<double *>(data.get()), ndims, d);
    }
    else if (datatype.template is_same<long double>())
    {
        printNdArray(os, reinterpret_cast<long double *>(data.get()), ndims, d);
    }
    else if (datatype.template is_same<std::complex<double>>())
    {
        printNdArray(os, reinterpret_cast<std::complex<double> *>(data.get()),
                     ndims, d);
    }
    else if (datatype.template is_same<std::complex<float>>())
    {
        printNdArray(os, reinterpret_cast<std::complex<float> *>(data.get()),
                     ndims, d);
    }
    else
    {
        UNIMPLEMENTED2("Cannot print datatype:" + datatype.name());
    }

    return os;
}

namespace traits
{

DataSet make_dataset(DataType const &dtype, std::shared_ptr<void> const &p, size_t rank,
                     size_t const *dims)
{
    DataSet res;

    res.datatype = dtype;

    if (dims != nullptr)
    {
        res.dataspace = DataSpace::create_simple(static_cast<int>(rank), dims);
        res.memory_space = res.dataspace;
        if (GLOBAL_COMM.is_valid())
        {   //fixme calculate distributed array dimensions
            UNIMPLEMENTED2("fixme calculate distributed array dimensions");
        }
    }
    else
    {
        size_t count = rank;
        size_t offset = 0;
        size_t total_count = count;

        std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));

        res.dataspace = DataSpace::create_simple(1, &total_count);
        res.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);
        res.memory_space = DataSpace::create_simple(1, &count);

    }


    res.data = p;

    return std::move(res);
}

DataSet make_dataset(DataType const &dtype)
{
    DataSet res;

    res.datatype = dtype;

    res.data = nullptr;

    return std::move(res);
}
} // namespace traits
} // namespace simpla

