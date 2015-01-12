/**
 * @file data_set.cpp
 *
 *  Created on: 2014年12月12日
 *      Author: salmon
 */

namespace simpla
{

bool DataSet::sync(size_t flag)
{
	return dataspace.sync(data, datatype, flag);
}

}  // namespace simpla

