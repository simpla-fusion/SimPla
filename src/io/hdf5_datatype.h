/*
 * hdf5_datatype.h
 *
 *  created on: 2014-5-26
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
#include "../utilities/ntuple.h"
#include "../utilities/log.h"
namespace simpla
{
/** \ingroup HDF5
 *
 *  \brief HDF5 data type factory
 *
 *  @todo transform to DataType interface
 */
struct HDF5DataTypeFactory: public Factory<size_t, hid_t>
{
	typedef Factory<size_t, hid_t> base_type;

	HDF5DataTypeFactory();

	~HDF5DataTypeFactory();

	product_type create(identifier_type const &id) const
	{
		return base_type::create(id);
	}

	product_type create(std::type_info const &t_info) const
	{
		return base_type::create(std::type_index(t_info).hash_code());
	}

	template<typename T>
	product_type create() const
	{
		return base_type::create(hash<T>());
	}

	template<typename T> bool Register(std::string const &desc)
	{
		create_fun_callback callback = [desc]()->product_type
		{
			return H5LTtext_to_dtype(desc.c_str(),H5LT_DDL);
		};

		return Register<T>(callback).second;
	}

private:
	template<typename T> identifier_type hash()
	{
		return (std::type_index(typeid(T)).hash_code());
	}
public:

	template<typename T> auto Register(create_fun_callback const &callback)
	DECL_RET_TYPE(base_type::Register(std::make_pair(hash<T>(), callback)))

	template<typename T> bool Unregister()
	{
		return base_type::Unregister(hash<T>());
	}

}
;

#define GLOBAL_HDF5_DATA_TYPE_FACTORY  SingletonHolder<HDF5DataTypeFactory> ::instance()

}
// namespace simpla

#endif /* HDF5_DATATYPE_H_ */
