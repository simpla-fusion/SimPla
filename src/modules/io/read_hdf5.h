/*
 * read_hdf5.h
 *
 *  Created on: 2012-10-28
 *      Author: salmon
 */

#ifndef READ_HDF5_H_
#define READ_HDF5_H_
#include <algorithm>
#include "include/simpla_defs.h"
#include "engine/arrayobject.h"
namespace simpla
{
namespace io
{
void ReadData(std::string const & name, TR1::shared_ptr<ArrayObject> obj);
}  // namespace io

}  // namespace simpla

#endif /* READ_HDF5_H_ */
