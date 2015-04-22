/**
 * @file dataset.cpp
 *
 *  Created on: 2014年12月12日
 *      Author: salmon
 */

#include "dataset.h"
#include "../utilities/utilities.h"
#include <ostream>
namespace simpla
{

std::ostream & DataSet::print(std::ostream & os) const
{
	auto shape = dataspace.local_shape();
	int ndims = 0;

	size_t d[shape.ndims];
	for (int i = 0; i < shape.ndims; ++i)
	{
		if (shape.dimensions[i] > 1)
		{
			d[ndims] = shape.dimensions[i];
			++ndims;
		}

	}

	if (datatype.is_same<int>())
	{
		printNdArray(os, reinterpret_cast<int*>(data.get()), ndims, d);
	}
	else if (datatype.is_same<long>())
	{
		printNdArray(os, reinterpret_cast<long*>(data.get()), ndims, d);
	}
	else if (datatype.is_same<unsigned long>())
	{
		printNdArray(os, reinterpret_cast<unsigned long*>(data.get()), ndims,
				d);
	}
	else if (datatype.is_same<float>())
	{
		printNdArray(os, reinterpret_cast<float*>(data.get()), ndims, d);
	}
	else if (datatype.is_same<double>())
	{
		printNdArray(os, reinterpret_cast<double*>(data.get()), ndims, d);
	}
	else if (datatype.is_same<long double>())
	{
		printNdArray(os, reinterpret_cast<long double*>(data.get()), ndims, d);
	}
	else if (datatype.is_same<std::complex<double>>())
	{
		printNdArray(os, reinterpret_cast<std::complex<double>*>(data.get()),
				ndims, d);
	}
	else if (datatype.is_same<std::complex<float>>())
	{
		printNdArray(os, reinterpret_cast<std::complex<float>*>(data.get()),
				ndims, d);
	}
	else
	{
		UNIMPLEMENTED2("Cannot print datatype:" + datatype.name());
	}

	return os;
}
}  // namespace simpla

