/*
 * read_hdf5.cpp
 *
 *  Created on: 2012-10-28
 *      Author: salmon
 */

#include "read_hdf5.h"
#include <H5Cpp.h>
#include <hdf5_hl.h>

void simpla::io::ReadData(std::string const &url, TR1::shared_ptr<ArrayObject> obj)
{

	int domain_pos = url.find(":", 0);
	int filename_pos = url.find(":", domain_pos + 1);
	std::string filename = url.substr(domain_pos + 3,
			filename_pos - domain_pos - 3);
	std::string path = url.substr(filename_pos + 1);

	H5::DataSet dataset = H5::H5File(filename, H5F_ACC_RDONLY).openDataSet(
			path);

	int nd = obj->get_num_of_dimesnsions();
	hsize_t dims[nd];
	obj->get_dimensions(dims);

	dataset.read(obj->get_data(),
			H5LTtext_to_dtype(obj->get_element_type_desc().c_str(), H5LT_DDL),
			H5::DataSpace(nd, dims));

}
