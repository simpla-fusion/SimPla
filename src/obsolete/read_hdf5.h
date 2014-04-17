/*
 * read_hdf5.h
 *
 *  Created on: 2012-10-28
 *      Author: salmon
 */

#ifndef READ_HDF5_H_
#define READ_HDF5_H_

#include <string>
#include <vector>

#include "include/simpla_defs.h"

namespace simpla
{
template<typename TV>
void HDF5Read(std::string const & name, std::vector<TV> *v)
{

}

template<typename TV>
inline void HDF5Read(H5::Group & grp, std::string const & name,
		std::vector<TV> *v)
{

}

}  // namespace simpla

#endif /* READ_HDF5_H_ */
