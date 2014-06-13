/*
 * hdf5_datatype.h
 *
 *  Created on: 2014年5月26日
 *      Author: salmon
 */

#ifndef HDF5_DATATYPE_H_
#define HDF5_DATATYPE_H_
extern "C"
{
#include <hdf5.h>
#include <hdf5_hl.h>
}
#include <complex>
#include <utility>
#include <typeindex>
#include <map>
#include <functional>

#include "../utilities/utilities.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/factory.h"
#include "../fetl/ntuple.h"

namespace simpla
{

struct HDF5DataTypeFactory: public Factory<size_t, hid_t>
{
	typedef Factory<size_t, hid_t> base_type;

	typedef std::function<hid_t()> create_fun_callback;

	HDF5DataTypeFactory();

	~HDF5DataTypeFactory();

	template<typename T>
	hid_t Create() const
	{
		return base_type::Create(hash<T>());
	}

	hid_t Create(size_t idx) const
	{
		return base_type::Create(idx);
	}

	template<typename T> bool Register(std::string const &desc)
	{
		create_fun_callback callback = [desc]()->hid_t
		{
			return H5LTtext_to_dtype(desc.c_str(),H5LT_DDL);
		};

		return Register<T>(callback);
	}

	template<typename T> bool Register(create_fun_callback const &callback)
	{
		return base_type::Register(hash<T>(), callback);;

	}
	template<typename T> bool Unregister()
	{
		return base_type::Unregister(hash<T>());
	}

private:
	template<typename T> auto hash()
	DECL_RET_TYPE((std::type_index(typeid(T)).hash_code()))

}
;

#define GLOBAL_HDF5_DATA_TYPE_FACTORY  SingletonHolder<HDF5DataTypeFactory> ::instance()

}
// namespace simpla

#endif /* HDF5_DATATYPE_H_ */
