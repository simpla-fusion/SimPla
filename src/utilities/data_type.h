/*
 * data_type.h
 *
 *  Created on: 2014年6月2日
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_
#include <string>
namespace simpla
{

struct DataTypeDesc
{

	int array_length = 1;
	std::string type_name, sub_type_name;
};

template<typename T>
class DataType
{
public:
	DataTypeDesc desc_;
	DataTypeDesc const &Desc() const
	{
		return desc_;
	}
};

}  // namespace simpla

#endif /* DATA_TYPE_H_ */
