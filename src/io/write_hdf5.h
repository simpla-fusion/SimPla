/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * IO/WriteHDF5.h
 *
 *  Created on: 2011-1-27
 *      Author: salmon
 */

#ifndef SRC_IO_WRITE_HDF5_H_
#define SRC_IO_WRITE_HDF5_H_
#include <H5Cpp.h>
#include "include/simpla_defs.h"
#include "engine/object.h"

namespace simpla
{
namespace io
{
void HDF5Append(H5::Group grp, std::string const & name,
		const NdArray::Holder obj);

void WriteHDF5(H5::Group grp, std::string const & name, NdArray const & obj);


NdArray::Holder ReadHDF5(H5::Group grp, std::string const & name);

void HDF5AddAttribute(H5::DataSet dataset, const NdArray::Holder obj);

} // namespace IO
} // namespace simpla
#endif  // SRC_IO_WRITE_HDF5_H_
