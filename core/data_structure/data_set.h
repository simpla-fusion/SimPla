/*
 * data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SET_H_
#define CORE_DATA_STRUCTURE_DATA_SET_H_
#include "data_space.h"
#include "data_type.h"
namespace simpla
{
struct DataType;
struct DataSpace;

struct DataSet
{

	std::shared_ptr<void> data;
	DataType datatype;
	DataSpace dataspace;
};
}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SET_H_ */
