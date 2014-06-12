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
	Init();
}
HDF5DataTypeFactory::~HDF5DataTypeFactory()
{

}
void HDF5DataTypeFactory::Init()
{

	factory_[std::type_index(typeid(long)).hash_code()] = []()->hid_t
	{	return H5T_NATIVE_LONG;};

	factory_[std::type_index(typeid(int)).hash_code()] = []()->hid_t
	{	return H5T_NATIVE_INT;};

	factory_[std::type_index(typeid(unsigned long)).hash_code()] = []()->hid_t
	{	return H5T_NATIVE_ULONG;};

	factory_[std::type_index(typeid(float)).hash_code()] = []()->hid_t
	{	return H5T_NATIVE_FLOAT;};

	factory_[std::type_index(typeid(double)).hash_code()] = []()->hid_t
	{	return H5T_NATIVE_DOUBLE;};

	factory_[std::type_index(typeid(std::complex<double>)).hash_code()] = []()->hid_t
	{
		hid_t type_ =

		H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
		H5Tinsert(type_, "r", 0, H5T_NATIVE_DOUBLE);
		H5Tinsert(type_, "i", sizeof(double), H5T_NATIVE_DOUBLE);

		return type_;
	};

}
hid_t HDF5DataTypeFactory::Create(size_t t_idx_) const
{
	hid_t res;

	try
	{
		res = factory_.at(t_idx_)();

	} catch (std::out_of_range const &)
	{
		RUNTIME_ERROR("unknown type!");
	}
	return res;
}
}  // namespace simpla
