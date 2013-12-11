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

#include <cstddef>
#include <memory>
#include <string>

extern "C"
{
#include <hdf5.h>
}

namespace simpla
{

namespace HDF5
{

class H5OutStream
{
	hid_t h5_file_;
	hid_t grp_;
	std::string ds_name_;
	std::string default_ds_name_;
	size_t write_count_;
	bool append_enabled_;
	bool dims_setted_;
	std::vector<size_t> dims_;
public:
	H5OutStream(std::string const & filename, std::string const & dsname =
			"unnamed") :

			default_ds_name_(dsname), write_count_(0),

			append_enabled_(false), dims_setted_(false)
	{
		h5_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
		H5P_DEFAULT);

		grp_ = H5Gcreate(h5_file_, "/", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	}

	~H5OutStream()
	{
		H5Gclose(grp_);
		H5Fclose(h5_file_);
	}
	inline void SetAppendEnabled(bool flag)
	{
		append_enabled_ = flag;
	}
	void OpenGroup(std::string const & name)
	{
		if (name[0] == '/') /// absolute path
		{
			H5Gclose(grp_);
			grp_ = H5Gopen(h5_file_, "/", H5P_DEFAULT);
		}
		else
		{
			hid_t fg = H5Gopen(grp_, name.c_str(), H5P_DEFAULT);
			H5Gclose(grp_);
			grp_ = fg;
		}

	}
	void OpenDataSet(std::string const & name)
	{
		ds_name_ = name;
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
