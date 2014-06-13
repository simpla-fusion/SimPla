/*
 * hdf5_datatype.cpp
 *
 *  Created on: 2014年6月3日
 *      Author: salmon
 */

#include <typeindex>
#include "hdf5_datatype.h"

namespace simpla
{
HDF5DataTypeFactory::HDF5DataTypeFactory()
{
	Register<long>([]()->hid_t
	{	return H5T_NATIVE_LONG;});

	Register<int>([]()->hid_t
	{	return H5T_NATIVE_INT;});

	Register<unsigned long>([]()->hid_t
	{	return H5T_NATIVE_ULONG;});

	Register<float>([]()->hid_t
	{	return H5T_NATIVE_FLOAT;});

	Register<double>([]()->hid_t
	{	return H5T_NATIVE_DOUBLE;});

	Register<std::complex<double>>([]()->hid_t
	{
		hid_t type_ =

		H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
		H5Tinsert(type_, "r", 0, H5T_NATIVE_DOUBLE);
		H5Tinsert(type_, "i", sizeof(double), H5T_NATIVE_DOUBLE);

		return type_;
	});
}
HDF5DataTypeFactory::~HDF5DataTypeFactory()
{

}

}  // namespace simpla
