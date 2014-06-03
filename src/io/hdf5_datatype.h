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
#include <unordered_map>

#include "../utilities/utilities.h"
#include "../utilities/singleton_holder.h"
#include "../fetl/ntuple.h"

namespace simpla
{

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){ H5Eprint(H5E_DEFAULT, stderr);}

struct HDF5DataTypeFactory
{
	HDF5DataTypeFactory();

	~HDF5DataTypeFactory();

	void Init();

	hid_t Create(std::type_index const & t_idx_) const;

	typedef std::function<hid_t()> create_fun_type;

	template<typename T> void Register(create_fun_type const &fun)
	{
		factory_[std::type_index(typeid(T))] = fun;
	}
	template<typename T> void Unegister(create_fun_type const &fun)
	{
		factory_.erase(std::type_index(typeid(T)));
	}

	std::unordered_map<std::type_index, create_fun_type> factory_;
}
;

#define GLOBAL_HDF5_DATA_TYPE_FACTORY  SingletonHolder<HDF5DataTypeFactory> ::instance()

}
// namespace simpla

#endif /* HDF5_DATATYPE_H_ */
