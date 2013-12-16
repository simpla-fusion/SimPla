/*
 * data_stream.h
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
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

#define H5_ERROR( _FUN_ ) if((_FUN_)<0){ /*H5Eprint(H5E_DEFAULT, stderr);*/}

class DataStream: public SingletonHolder<DataStream>
{
	std::string prefix_;
	int suffix_width_;

	std::string filename_;
	std::string grpname_;
	hid_t file_;
	hid_t group_;

public:

	DataStream() :
			prefix_("simpla_unnamed"), filename_("unnamed"), grpname_(""), file_(
					-1), group_(-1), suffix_width_(4)
	{
		hid_t error_stack = H5Eget_current_stack();
		H5Eset_auto(error_stack, NULL, NULL);
	}

	~DataStream()
	{
		CloseFile();
	}

	inline void OpenGroup(std::string const & gname)
	{
		hid_t h5fg = file_;
		CloseGroup();
		if (gname[0] == '/')
		{
			grpname_ = gname;
		}
		else
		{
			grpname_ += gname;
			if (group_ > 0)
				h5fg = group_;
		}

		if (grpname_[grpname_.size() - 1] != '/')
		{
			grpname_ = grpname_ + "/";
		}

		auto res = H5Lexists(h5fg, grpname_.c_str(), H5P_DEFAULT);

		if (grpname_ == "/" || res != 0)
		{
			H5_ERROR(group_ = H5Gopen(h5fg, grpname_.c_str(), H5P_DEFAULT));
		}
		else
		{
			H5_ERROR(
					group_ = H5Gcreate(h5fg, grpname_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
		}
		if (group_ <= 0)
		{
			ERROR << "Can not open group " << grpname_ << " in file "
					<< prefix_;
		}

	}

	inline void OpenFile(std::string const &fname = "unnamed")
	{

		CloseFile();
		if (fname != "")
			prefix_ = fname;

		if (fname.size() > 3 && fname.substr(fname.size() - 3) == ".h5")
		{
			prefix_ = fname.substr(0, fname.size() - 3);
		}

		/// @TODO auto mkdir directory

		filename_ = prefix_ +

		AutoIncrease(

		[&](std::string const & suffix)->bool
		{
			std::string fname=(prefix_+suffix);
			return
			fname==""
			|| *(fname.rbegin())=='/'
			|| (CheckFileExists(fname + ".h5"));
		}

		) + ".h5";

		H5_ERROR(
				file_ = H5Fcreate(filename_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
		if (file_ < 0)
		{
			ERROR << "Create HDF5 file " << filename_ << " failed!"
					<< std::endl;
		}
		OpenGroup("");
	}

	void CloseGroup()
	{
		if (group_ > 0)
		{
			H5Gclose(group_);
		}
		group_ = -1;
	}
	void CloseFile()
	{
		CloseGroup();
		if (file_ > 0)
		{
			H5Fclose(file_);
		}
		file_ = -1;
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

	template<typename ...Args>
	std::string Write(Args const &... args) const
	{
		return HDF5Write(group_, std::forward<Args const&>(args)...);
	}

}
;

#define GLOBAL_DATA_STREAM DataStream::instance()

template<typename TV>
class DataSet
{

	std::shared_ptr<TV> data_;
	std::string name_;
	bool is_compact_store_;
	std::vector<size_t> dims_;
public:

	typedef TV value_type;

	static const size_t LIGHT_DATA_LIMIT = 256;

	DataSet(std::shared_ptr<TV> const & d, std::string const &name = "unnamed",
			int rank = 1, size_t const* dims = nullptr, bool flag = false) :
			data_(d), name_(name), is_compact_store_(flag)
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
	DataSet(std::shared_ptr<TV> const & d, std::string const &name,
			nTuple<N, TI> const & dims, bool flag = false) :
			data_(d), name_(name), is_compact_store_(flag)
	{
		for (size_t i = 0; i < N; ++i)
		{
			dims_.push_back(dims[i]);
		}
	}

	DataSet(std::shared_ptr<TV> const & d, std::string const &name,
			std::vector<size_t> const & dims, bool flag = false) :
			data_(d), name_(name), dims_(dims), is_compact_store_(flag)
	{
	}

	DataSet(DataSet && r) :
			data_(r.data_), name_(r.name_), is_compact_store_(
					r.is_compact_store_), dims_(r.dims_)
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
	bool IsHeavyData() const
	{
		return size() > LIGHT_DATA_LIMIT;
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

template<typename TV, typename ... Args> inline DataSet<TV> Data(
		std::shared_ptr<TV> const & d, Args const & ... args)
{
	return std::move(DataSet<TV>(d, std::forward<Args const &>(args)...));
}

template<typename U>
std::ostream & operator<<(std::ostream & os, DataSet<U> const &d)
{
	if (!d.IsHeavyData() && (!d.IsCompactStored()))
	{
		PrintNdArray(os, d.get(), d.GetDims().size(), &(d.GetDims()[0]));
	}
	else
	{

		os

		<< "\""

		<< GLOBAL_DATA_STREAM.GetCurrentPath()

		<< GLOBAL_DATA_STREAM.Write(d)

		<< "\"";
	}

	return os;

}

template<typename T> struct HDF5DataType
{
	hid_t type() const
	{
		return H5T_NATIVE_INT;
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

std::string HDF5Write(hid_t grp, void const *v, std::string const &name,
		hid_t mdtype, int rank, size_t const *dims, bool is_apppendable);

template<typename TV>
std::string HDF5Write(hid_t grp, TV const *v, std::string const &name,
		std::vector<size_t> const &d, bool APPEND)
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

//	auto mdtype = HDF5DataType<typename nTupleTraits<TV>::value_type>();

	std::string res = HDF5Write(grp, reinterpret_cast<void const*>(v), name,

	HDF5DataType<typename nTupleTraits<TV>::value_type>().type(),

	dims.size(), &dims[0], APPEND);

	return res;

}

template<typename U, typename ... Args>
std::string HDF5Write(hid_t grp, DataSet<U> const & d, Args const &... args)
{

	return std::move(
			HDF5Write(grp, d.get(), d.GetName(), d.GetDims(), d.IsAppendable()),
			std::forward<Args const &>(args)...);
}
}
// namespace simpla

#endif /* DATA_STREAM_ */
