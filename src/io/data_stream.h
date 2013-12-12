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

#include "../fetl/ntuple_ops.h"
#include "../fetl/ntuple.h"
#include "../utilities/log.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/utilities.h"
#include "../utilities/pertty_stream.h"

namespace simpla
{
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
			prefix_("simpla_unnamed"), file_(0), group_(0), suffix_width_(4)
	{
	}

	~DataStream()
	{
		CloseFile();
	}

	inline void OpenGroup(std::string const & gname)
	{
		CloseGroup();

		grpname_ = gname + "/";

		group_ = H5Gopen(file_, grpname_.c_str(), H5P_DEFAULT);

		if (group_ <= 0)
		{
			group_ = H5Gcreate(file_, grpname_.c_str(), H5P_DEFAULT,
			H5P_DEFAULT, H5P_DEFAULT);
		}
		if (group_ <= 0)
		{
			ERROR << "Can not open group " << grpname_ << " in file "
					<< prefix_;
		}

	}

	inline void OpenFile(std::string const &fname = "")
	{

		CloseFile();
		if (fname != "")
			prefix_ = fname;

		if (fname.size() > 3 && fname.substr(fname.size() - 3) == ".h5")
		{
			prefix_ = fname.substr(0, fname.size() - 3);
		}

		filename_ = prefix_ +

		AutoIncrease(

		[&](std::string const & suffix)->bool
		{
			file_ = H5Fcreate((prefix_+suffix+".h5").c_str(),
					H5F_ACC_EXCL, H5P_DEFAULT,
					H5P_DEFAULT );
			return file_>0;
		}

		) + ".h5";

		OpenGroup("");
	}

	void CloseGroup()
	{
		if (group_ > 0)
		{
			H5Gclose(group_);
		}
		group_ = 0;
		grpname_ = "";
	}
	void CloseFile()
	{
		CloseGroup();
		if (file_ > 0)
		{
			H5Fclose(file_);
		}
		file_ = 0;
		filename_ = "";
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
		return HDF5Write(group_, std::forward<Args const &>(args)...);
	}

};

#define GLOBAL_DATA_STREAM DataStream::instance()

template<typename T>
class DataSet
{

	T const & data_;
	std::string name_;
	bool appendable_;
	std::vector<size_t> dims_;
public:

	static const size_t LIGHT_DATA_LIMIT = 256;

	DataSet(T const & d, std::string const &name = "unnamed", int rank = 1,
			size_t * dims = nullptr) :
			data_(d), name_(name), appendable_(false)
	{
		if (dims != nullptr && rank > 0)
		{
			for (size_t i = 0; i < rank; ++i)
			{
				dims_.push_back(dims[i]);
			}

		}
	}
	DataSet(DataSet && r) :
			data_(r.data_), name_(r.name_), appendable_(r.appendable_), dims_(
					r.dims_)
	{

	}
	~DataSet()
	{

	}

	bool IsHeavyData() const
	{
		return data_.size() > LIGHT_DATA_LIMIT;
	}
	inline T const & GetData() const
	{
		return data_;
	}

	bool IsAppendable() const
	{
		return appendable_;
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

template<typename T, typename ... Args> inline DataSet<T> Data(T const & d,
		Args const & ... args)
{
	return std::move(DataSet<T>(d, std::forward<Args const &>(args)...));
}

template<typename U>
std::ostream & operator<<(std::ostream & os, DataSet<U> const &d)
{
	int rank = 0;
	size_t const * dims = nullptr;
	if (!d.GetDims().empty())
	{
		rank = d.GetDims().size();
		dims = &d.GetDims()[0];
	}

	if (!d.IsHeavyData())
	{
		os << "{" << d.GetData() << "}";
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

template<typename U>
std::string HDF5Write(hid_t grp, DataSet<U> const & d)
{
	return std::move(
			HDF5Write(grp, d.GetData(), d.GetName(), d.GetDims(),
					d.IsAppendable()));
}

template<typename TV, typename ...TOther>
std::string HDF5Write(hid_t grp, std::vector<TV, TOther...> const &v,
		std::string const &name, std::vector<size_t> d, bool is_apppendable =
				false)
{
	if (grp <= 0)
	{
		WARNING << "HDF5 file is not opened! No data is saved!";
		return "";
	}

	int rank = d.size();

	size_t dims[rank + 2];

	std::copy(d.begin(), d.end(), dims);

	if (is_nTuple<TV>::value)
	{
		dims[rank] = nTupleTraits<TV>::NUM_OF_DIMS;
		++rank;
	}

	hid_t mdtype = HDF5DataType<typename nTupleTraits<TV>::value_type>().type();

	std::string dsname = name +

	AutoIncrease([&](std::string const & s )->bool
	{
		return H5Gget_objinfo(grp,
				(name + s ).c_str(), false, nullptr) < 0;
	}, 0, 4);

	if (!is_apppendable)
	{

		hsize_t mdims[rank];

		std::copy(dims, dims + rank, mdims);

		hid_t dspace = H5Screate_simple(rank, mdims, mdims);
		hid_t dset = H5Dcreate(grp, dsname.c_str(), mdtype, dspace, H5P_DEFAULT,
		H5P_DEFAULT, H5P_DEFAULT);

		H5Dwrite(dset, mdtype, dspace, dspace, H5P_DEFAULT,
				static_cast<void const*>(&v[0]));

		H5Dclose(dset);
		H5Dclose(dspace);

	}
	else
	{
		hid_t dset;
		hid_t fspace;
		hid_t mspace;

		int ndims = rank + 1;

		hsize_t start[ndims];
		std::fill(start, start + ndims, 0);

		hsize_t mdims[ndims];
		mdims[0] = 1;
		std::copy(dims, dims + rank, mdims + 1);
		hsize_t fdims[ndims];
		fdims[0] = H5S_UNLIMITED;
		std::copy(dims, dims + rank, fdims + 1);

		mspace = H5Screate_simple(ndims, mdims, mdims);

		if (H5LTfind_dataset(grp, dsname.c_str()))
		{
			dset = H5Dopen1(grp, dsname.c_str());
			fspace = H5Dget_space(dset);
			H5Sset_extent_simple(fspace, ndims, fdims, fdims);
		}
		else
		{
			hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
			H5Pset_chunk(plist, ndims, mdims);
			fspace = H5Screate_simple(ndims, mdims, fdims);
			dset = H5Dcreate(grp, dsname.c_str(), mdtype, fspace, plist,
			H5P_DEFAULT, H5P_DEFAULT);
			fdims[0] = 0;
			H5Pclose(plist);
		}

		start[0] = fdims[0];

		++fdims[0];

		H5Dextend(dset, fdims);

		H5Fflush(grp, H5F_SCOPE_GLOBAL);

		fspace = H5Dget_space(dset);

		H5Sselect_hyperslab(fspace, H5S_SELECT_SET, mdims, start, H5P_DEFAULT,
		H5P_DEFAULT);

		H5Dwrite(dset, mdtype, mspace, fspace, H5P_DEFAULT,
				static_cast<void const*>(&v[0]));

		H5Sclose(mspace);
		H5Sclose(fspace);
		H5Dclose(dset);
	}

	return dsname;

}

}
// namespace simpla

#endif /* DATA_STREAM_ */
