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

template<typename TV>
struct DataType<std::complex<TV> > : public BaseDataType
{
	DataType()
	{
	}
	virtual ~DataType()
	{
	}
	virtual size_t size_in_bytes() const
	{
		return DataType<TV>().size_in_bytes() * 2;
	}

	virtual std::string desc() const
	{
		std::ostringstream os;
		os << "H5T_COMPOUND {  "

		<< DataType<TV>().desc() << "  \"r\" : " << offsetof(_complex, r) << ";"

		<< DataType<TV>().desc() << "  \"i\" : " << offsetof(_complex, i) << ";"

		<< "}";

		return os.str();
	}
private:
	struct _complex
	{
		TV r;
		TV i;
	};
};
template<int N, typename TV> class nTuple;

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
