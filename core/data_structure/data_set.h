/*
 * data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SET_H_
#define CORE_DATA_STRUCTURE_DATA_SET_H_
#include "data_type.h"
#include "dataspace.h"
namespace simpla
{
struct DataType;
struct DataSpace;
struct Properties;

struct DataSet
{
	Properties attribute;
	std::shared_ptr<void> data;
	DataType datatype;
	DataSpace dataspace;
};

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SET_H_ */
