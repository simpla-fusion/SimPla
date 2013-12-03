/*
 * hdf5_stream.h
 *
 *  Created on: 2013年12月3日
 *      Author: salmon
 */

#ifndef HDF5_STREAM_H_
#define HDF5_STREAM_H_

#include "write_hdf5.h"
#include "../utilities/utilities.h"
#include <H5Cpp.h>
#include <cstddef>
#include <memory>
#include <string>

namespace simpla
{

namespace HDF5
{

class H5OutStream
{
	H5::H5File h5_file_;
	H5::Group grp_;
	H5::DataSet ds_;
	H5::DataSpace dspace_;
	std::string ds_name_;
	std::string default_ds_name_;
	size_t write_count_;
	bool append_enabled_;
	bool dims_setted_;
	std::vector<size_t> dims_;
public:
	H5OutStream(std::string const & filename, std::string const & dsname =
			"unnamed") :
			h5_file_(filename, H5F_ACC_TRUNC), grp_(h5_file_.openGroup("/")),

			default_ds_name_(dsname),

			write_count_(0),

			append_enabled_(false),

			dims_setted_(false)
	{
	}
	inline void SetAppendEnabled(bool flag)
	{
		append_enabled_ = flag;
	}
	void OpenGroup(std::string const & name)
	{
		H5::Group fg = grp_;

		if (name[0] == '/') /// absolute path
		{
			fg = h5_file_.openGroup("/");
		}

		try
		{
			grp_ = fg.openGroup(name.c_str());
		} catch (...)
		{
			grp_ = fg.createGroup(name.c_str());
		}
	}
	void OpenDataSet(std::string const & name)
	{
		ds_name_ = name;
//		append_enabled_ = true;
	}
	void CloseDataSet()
	{
		ds_name_ = "";
		append_enabled_ = false;
		ClearDims();
	}

	template<typename T>
	void SetDims(std::vector<T> const & d)
	{
		dims_.clear();
		dims_.insert(dims_.begin(), d.begin(), d.end());
	}

	void ClearDims()
	{
		dims_.clear();
	}

	template<typename T>
	void Write(T const &v)
	{
		if (ds_name_ == "")
		{
			ds_name_ = default_ds_name_ + ToString(write_count_);
			++write_count_;
		}

		HDF5Write(grp_, ds_name_, v, dims_.size(), &dims_[0], append_enabled_);

	}

}
;

struct OpenGroup
{
	std::string name_;
};

inline H5OutStream & operator<<(H5OutStream & h5os, OpenGroup const & ogrp)
{
	h5os.OpenGroup(ogrp.name_);
	return h5os;
}

struct OpenDataSet
{
	template<typename T>
	OpenDataSet(T const & n) :
			name_(n)
	{
	}
	std::string name_;
};

struct CloseDataSet
{
};

inline H5OutStream & operator<<(H5OutStream & h5os,
		OpenDataSet const & OpenDataSet)
{
	h5os.OpenDataSet(OpenDataSet.name_);
	return h5os;
}
inline H5OutStream & operator<<(H5OutStream & h5os,
		CloseDataSet const & OpenDataSet)
{
	h5os.CloseDataSet();
	return h5os;
}

struct SetDims
{
	std::vector<size_t> d_;
	template<typename T>
	SetDims(int n, T const* d) :
			d_(d, d + n)
	{
	}

	template<int N, typename T>
	SetDims(nTuple<N, T> const & d) :
			d_(&d[0], &d[0] + N)
	{
	}

	template<typename T>
	SetDims(T const & d) :
			d_(d.begin(), d.end())
	{
	}

};
struct ClearDims
{
};
inline H5OutStream & operator<<(H5OutStream & h5os, SetDims const &op)
{
	h5os.SetDims(op.d_);
	return h5os;
}
inline H5OutStream & operator<<(H5OutStream & h5os, ClearDims const &)
{
	h5os.ClearDims();
	return h5os;
}
struct EnableAppend
{
};
struct DisableAppend
{
};
inline H5OutStream & operator<<(H5OutStream & h5os, EnableAppend const &)
{
	h5os.SetAppendEnabled(true);
	return h5os;
}
inline H5OutStream & operator<<(H5OutStream & h5os, DisableAppend const &)
{
	h5os.SetAppendEnabled(false);
	return h5os;
}
template<typename T>
inline H5OutStream & operator<<(H5OutStream & h5os, T const &v)
{
	h5os.Write(v);
	return h5os;
}

} // namespace HDF5
}  // namespace simpla

#endif /* HDF5_STREAM_H_ */
