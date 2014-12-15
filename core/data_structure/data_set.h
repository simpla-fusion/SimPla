/*
 * data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SET_H_
#define CORE_DATA_STRUCTURE_DATA_SET_H_

#include <memory>

#include "../utilities/properties.h"
#include "data_type.h"
#include "data_space.h"

namespace simpla
{

/**
 *  @brief DataSet
 *
 *
 */

struct DataSet
{
	std::shared_ptr<void> data;
	Properties attribute;
	DataType datatype;
	DataSpace dataspace;

	bool is_valid() const
	{
		return data != nullptr && datatype.is_valid() && dataspace.is_valid();
	}
};

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SET_H_ */
