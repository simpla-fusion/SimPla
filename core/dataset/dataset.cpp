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
//		CHECK(datatype.size_in_byte() * dataspace.local_memory_size());
		data = SingletonHolder<MemoryPool>::instance().raw_alloc(
				datatype.size_in_byte() * dataspace.local_memory_size());
	}
}

void DataSet::clear()
{
	deploy();
	memset(data.get(), 0, dataspace.local_memory_size() * datatype.size_in_byte());
}


void DataSet::copy(void const *other)
{
	deploy();
	memcpy(data.get(), other, dataspace.local_memory_size() * datatype.size_in_byte());
}

bool DataSet::is_same(void const *other) const
{
	return data.get() == other;
}

bool DataSet::is_equal(void const *other) const
{
	return is_same(other) ||
			memcmp(data.get(), other, dataspace.local_memory_size() * datatype.size_in_byte()) != 0;
}

std::ostream &DataSet::print(std::ostream &os) const
{
	auto shape = dataspace.local_shape();

	int ndims = 0;

	long d[shape.ndims];

	for (int i = 0; i < shape.ndims; ++i)
	{
		if (shape.dimensions[i] > 1)
		{
			d[ndims] = shape.dimensions[i];
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
}  // namespace simpla

