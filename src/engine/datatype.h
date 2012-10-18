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
	BaseDataType()
	{
	}
	virtual ~BaseDataType()
	{
	}

	typedef TR1::shared_ptr<BaseDataType> Holder;

	virtual size_t size_in_bytes() const=0;
	virtual std::string desc() const=0;
};

template<typename T> struct DataType;
template<>
struct DataType<double> : public BaseDataType
{
	DataType()
	{
	}
	virtual ~DataType()
	{
	}
	virtual size_t size_in_bytes() const
	{
		return sizeof(double);
	}

	virtual std::string desc() const
	{
		return "H5T_NATIVE_DOUBLE";
	}
};

template<int N, typename TV>

struct DataType<nTuple<N, TV> > : public BaseDataType
{
	DataType()
	{
	}
	virtual ~DataType()
	{
	}
	virtual size_t size_in_bytes() const
	{
		return DataType<TV>().size_in_bytes() * N;
	}

	virtual std::string desc() const
	{
		std::ostringstream os;
		os << "   H5T_ARRAY { [" << N << "] " << DataType<TV>().desc() << "}  ";
		return os.str();
	}
};

}  // namespace simpla

#endif /* DATATYPE_H_ */
