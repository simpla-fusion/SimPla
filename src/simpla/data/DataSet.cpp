/**
 * @file DataSet.cpp
 *
 *  Created on: 2014-12-12
 *      Author: salmon
 */
#include <ostream>
#include <string.h>

#include "DataSet.h"
#include <simpla/toolbox/FancyStream.h>

namespace simpla { namespace data
{


bool DataSet::is_same(void const *other) const
{
    return data.get() == other;
}

bool DataSet::is_equal(void const *other) const
{
    return is_same(other) ||
           memcmp(data.get(), other, memory_space.size() * data_type.size_in_byte()) != 0;
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

    if (data_type.template is_same<int>())
    {
        printNdArray(os, reinterpret_cast<int *>(data.get()), ndims, d);
    }
    else if (data_type.template is_same<long>())
    {
        printNdArray(os, reinterpret_cast<long *>(data.get()), ndims, d);
    }
    else if (data_type.template is_same<unsigned long>())
    {
        printNdArray(os, reinterpret_cast<unsigned long *>(data.get()), ndims,
                     d);
    }
    else if (data_type.template is_same<float>())
    {
        printNdArray(os, reinterpret_cast<float *>(data.get()), ndims, d);
    }
    else if (data_type.template is_same<double>())
    {
        printNdArray(os, reinterpret_cast<double *>(data.get()), ndims, d);
    }
    else if (data_type.template is_same<long double>())
    {
        printNdArray(os, reinterpret_cast<long double *>(data.get()), ndims, d);
    }
    else if (data_type.template is_same<std::complex<double>>())
    {
        printNdArray(os, reinterpret_cast<std::complex<double> *>(data.get()),
                     ndims, d);
    }
    else if (data_type.template is_same<std::complex<float>>())
    {
        printNdArray(os, reinterpret_cast<std::complex<float> *>(data.get()),
                     ndims, d);
    }
    else
    {
        UNIMPLEMENTED2("Cannot print DataType:" + data_type.name());
    }

    return os;
}

}}//namespace simpla { namespace data_model

