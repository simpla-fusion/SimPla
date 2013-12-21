/*
 * data_stream.h
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 *
 *
 * TODO: DataStream and DataSet need improvement!!!
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

//#include <H5Epublic.h>
//#include <H5Ipublic.h>
//#include <H5LTpublic.h>
//#include <H5public.h>
//#include <H5Tpublic.h>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

extern "C"
{
#include <hdf5.h>
#include <hdf5_hl.h>

}

#include "../fetl/ntuple.h"
#include "../utilities/log.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/utilities.h"
#include "../utilities/pretty_stream.h"

namespace simpla
{

template<typename TV>
class DataSet
{

	std::shared_ptr<TV> data_;
	std::string name_;
	bool is_compact_store_;
	std::vector<size_t> dims_;
public:

	typedef TV value_type;

	DataSet(std::shared_ptr<TV> const & d, std::string const &name = "unnamed", int rank = 1, size_t const* dims =
	        nullptr, bool flag = false)
			: data_(d), name_(name), is_compact_store_(flag)
	{
		if (dims != nullptr && rank > 0)
		{
			for (size_t i = 0; i < rank; ++i)
			{
				dims_.push_back(dims[i]);
			}

		}
		else
		{
			ERROR << "Illegal input! [dims == nullptr or rank <=0] ";
		}

	}

	template<int N, typename TI>
	DataSet(std::shared_ptr<TV> const & d, std::string const &name, nTuple<N, TI> const & dims, bool flag = false)
			: data_(d), name_(name), is_compact_store_(flag)
	{
		for (size_t i = 0; i < N; ++i)
		{
			dims_.push_back(dims[i]);
		}
	}

	DataSet(std::shared_ptr<TV> const & d, std::string const &name, std::vector<size_t> const & dims, bool flag = false)
			: data_(d), name_(name), dims_(dims), is_compact_store_(flag)
	{
	}

	DataSet(DataSet && r)
			: data_(r.data_), name_(r.name_), is_compact_store_(r.is_compact_store_), dims_(r.dims_)
	{

	}
	~DataSet()
	{

	}
	inline size_t size() const
	{
		size_t s = 1;
		for (auto const &d : dims_)
		{
			s *= d;
		}
		return s;
	}

	bool IsCompactStored() const
	{
		return is_compact_store_;
	}
	inline const std::shared_ptr<value_type> data() const
	{
		return data_;
	}

	inline const value_type* get() const
	{
		return data_.get();
	}

	bool IsAppendable() const
	{
		return is_compact_store_;
	}

	const std::string& GetName() const
	{
		return name_;
	}
	const std::vector<size_t>& GetDims() const
	{
		return dims_;
	}
};

template<typename TV, typename ... Args> inline DataSet<TV> Data(std::shared_ptr<TV> const & d, Args const & ... args)
{
	return std::move(DataSet<TV>(d, std::forward<Args const &>(args)...));
}
template<typename TV, typename ... Args> inline DataSet<TV> Data(TV* d, Args const & ... args)
{
	return std::move(DataSet<TV>(d, std::forward<Args const &>(args)...));
}

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){ H5Eprint(H5E_DEFAULT, stderr);}

namespace _impl
{

HAS_STATIC_MEMBER_FUNCTION(DataTypeDesc);

template<typename T>
typename std::enable_if<has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetH5Type()
{
	hid_t res;
	H5_ERROR(res = H5LTtext_to_dtype(T::DataTypeDesc().c_str(), H5LT_DDL));
	return res;
}
template<typename T>
typename std::enable_if<!has_static_member_function_DataTypeDesc<T>::value, hid_t>::type GetH5Type()
{
	return H5T_OPAQUE;
}

}  // namespace _impl

template<typename T>
struct HDF5DataType
{
	hid_t type(...) const
	{
		return _impl::GetH5Type<T>();
	}
};

template<> struct HDF5DataType<int>
{
	hid_t type() const
	{
		return H5T_NATIVE_INT;
	}
};

template<> struct HDF5DataType<float>
{
	hid_t type() const
	{
		return H5T_NATIVE_FLOAT;
	}
};

template<> struct HDF5DataType<double>
{
	hid_t type() const
	{
		return H5T_NATIVE_DOUBLE;
	}
};

template<> struct HDF5DataType<long double>
{
	hid_t type() const
	{
		return H5T_NATIVE_LDOUBLE;
	}
};
template<typename T> struct HDF5DataType<std::complex<T>>
{
	hid_t type_;
	HDF5DataType()
	{
		type_ = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<T>));
		H5Tinsert(type_, "r", 0, HDF5DataType<T>().type());
		H5Tinsert(type_, "i", sizeof(T), HDF5DataType<T>().type());
	}

	~ HDF5DataType()
	{
		H5Tclose(type_);
	}

	hid_t type() const
	{
		return type_;
	}
};

std::string HDF5Write(hid_t grp, void const *v, std::string const &name, hid_t mdtype, int rank, size_t const *dims,
        bool is_apppendable);
template<typename ...Args>
std::string DataStreamWrite(Args const &... args)
{
	UNIMPLEMENT;
	return "";
}

class DataStream: public SingletonHolder<DataStream>
{
	std::string prefix_;
	int suffix_width_;

	std::string filename_;
	std::string grpname_;
	hid_t file_;
	hid_t group_;
	size_t LIGHT_DATA_LIMIT_;
	bool is_compact_storable_;
public:

	DataStream()
			: prefix_("simpla_unnamed"), filename_("unnamed"), grpname_(""),

			file_(-1), group_(-1),

			suffix_width_(4),

			LIGHT_DATA_LIMIT_(20),

			is_compact_storable_(true)

	{
		hid_t error_stack = H5Eget_current_stack();
		H5Eset_auto(error_stack, NULL, NULL);
	}

	~DataStream()
	{
		CloseFile();
	}
	void SetLightDatLimit(size_t s)
	{
		LIGHT_DATA_LIMIT_ = s;
	}
	size_t GetLightDatLimit() const
	{
		return LIGHT_DATA_LIMIT_;
	}

	void EnableCompactStorable()
	{
		is_compact_storable_ = true;
	}
	void DisableCompactStorable()
	{
		is_compact_storable_ = false;
	}

	bool CheckCompactStorable() const
	{
		return is_compact_storable_;
	}

	inline std::string GetCurrentPath() const
	{
		return filename_ + ":" + grpname_;
	}

	inline std::string GetPrefix() const
	{
		return prefix_;
	}

	inline void SetPrefix(const std::string& prefix)
	{
		prefix_ = prefix;
	}

	int GetSuffixWidth() const
	{
		return suffix_width_;
	}

	void SetSuffixWidth(int suffixWidth)
	{
		suffix_width_ = suffixWidth;
	}
	template<typename U>
	std::ostream & Serialize(std::ostream & os, DataSet<U> const & d)
	{
//		if (d.size() < LIGHT_DATA_LIMIT_ && !(d.IsCompactStored() && is_compact_storable_))
//		{
//			PrintNdArray(os, d.get(), d.GetDims().size(), &(d.GetDims()[0]));
//		}
//		else
		{
			os << "\"" << GetCurrentPath() << Write(d) << "\"";
		}
		return os;
	}

	void OpenGroup(std::string const & gname);
	void OpenFile(std::string const &fname = "unnamed");
	void CloseGroup();
	void CloseFile();

//	template<typename ...Args>
//	std::string Write(Args const &... args) const
//	{
//		return DataStreamWrite(group_, std::forward<Args const&>(args)...);
//	}

	template<typename U>
	std::string Write(DataSet<U> const & d) const
	{
		return std::move(Write(d.get(), d.GetName(), d.GetDims(), d.IsCompactStored()));
	}

	template<typename TV>
	std::string Write(TV const *v, std::string const &name, std::vector<size_t> const &d, bool is_compact_stored) const
	{

		if (v == nullptr)
		{
			WARNING << "empty data";
			return "empty data";
		}

		if (d.empty())
		{
			WARNING << "Unknown  size of dataset! ";
			return "Unknown  size of dataset";
		}

		std::vector<size_t> dims;

		std::vector<size_t>(d).swap(dims);

		if (is_nTuple<TV>::value)
		{
			dims.push_back(nTupleTraits<TV>::NUM_OF_DIMS);
		}

		std::string res = Write(reinterpret_cast<void const*>(v), name,

		HDF5DataType<typename nTupleTraits<TV>::value_type>().type(),

		dims.size(), &dims[0], is_compact_stored);

		return res;

	}

	std::string Write(void const *v, std::string const &name, hid_t mdtype, int rank, size_t const *dims,
	        bool is_compact_stored) const;

}
;

#define GLOBAL_DATA_STREAM DataStream::instance()

template<typename U>
std::ostream & operator<<(std::ostream & os, DataSet<U> const &d)
{
	DataStream::instance().Serialize(os, d);
	return os;
}

#define DUMP(_F_) Data(_F_,__STRING(_F_) ,true)
#ifndef NDEBUG
#	define DEBUG_DUMP(_F_) Data(_F_,__STRING(_F_),true)
#else
#   define DEBUG_DUMP(_F_) ""
#endif
}
// namespace simpla

#endif /* DATA_STREAM_ */
