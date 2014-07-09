/*
 * data_type.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_
#include <string>

namespace simpla
{

struct DataType
{
public:

	DataType(std::type_index const & t_idx)
			: t_idx_(t_idx)
	{
	}

	~DataType()
	{
	}

	template<typename TV> static DataType create()
	{
		return std::move(DataType(std::type_index(typeid(TV))));
	}

	int array_length = 1;
	std::type_index t_idx_;

};

}
// namespace simpla

#endif /* DATA_TYPE_H_ */
