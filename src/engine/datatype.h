/*
 * datatype.h
 *
 *  Created on: 2012-10-13
 *      Author: salmon
 */

#ifndef DATATYPE_H_
#define DATATYPE_H_
namespace simpla
{
class BaseDataType
{
public:
	BaseDataType(int es) :
			size(es)
	{
	}
	virtual ~BaseDataType()
	{
	}

	typedef TR1::shared_ptr<BaseDataType> Holder;

	const int size;

	virtual std::string H5DataTypeDesc()=0;
};

template<typename T> struct DataType;
template<>
struct DataType<double> : public BaseDataType
{
	DataType() :
			BaseDataType(sizeof(double))
	{
	}
	virtual ~DataType()
	{
	}
	virtual std::string H5DataTypeDesc() const
	{
		return "H5T_NATIVE_DOUBLE";
	}
};



}  // namespace simpla

#endif /* DATATYPE_H_ */
