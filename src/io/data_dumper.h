/*
 * data_dumper.h
 *
 *  Created on: 2014年5月7日
 *      Author: salmon
 */

#ifndef DATA_DUMPER_H_
#define DATA_DUMPER_H_

#include <stddef.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/ntuple.h"
#include "data_stream.h"

namespace simpla
{
template<typename TV>
class DataSaver
{

	TV const* data_;
	std::string name_;
	bool is_verbose_;
	std::vector<size_t> dims_;
public:

	typedef TV value_type;

	template<typename TI>
	DataSaver(TV const* d, std::string const &name = "unnamed", int rank = 1, TI const* dims = nullptr, bool flag =
	        false)
			: data_(d), name_(name), is_verbose_(flag)
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
	DataSaver(TV const* d, std::string const &name, nTuple<N, TI> const & dims, bool flag = false)
			: data_(d), name_(name), is_verbose_(flag)
	{
		for (size_t i = 0; i < N; ++i)
		{
			dims_.push_back(dims[i]);
		}
	}

	template<typename TI>
	DataSaver(TV const* d, std::string const &name, std::vector<TI> const & dims, bool flag = false)
			: data_(d), name_(name), dims_(dims.size()), is_verbose_(flag)
	{
		std::copy(dims.begin(), dims.end(), dims_.begin());
	}

	DataSaver(DataSaver const& r) = delete;

	DataSaver(DataSaver && r) = delete;

	~DataSaver()
	{
		GLOBAL_DATA_STREAM.Write(data_, name_, dims_.size(), &dims_[0], is_verbose_);
	}

	std::string GetName() const
	{
		return "\"" + GLOBAL_DATA_STREAM.GetCurrentPath() + name_ + "\"";
	}

};

template<typename TV, typename ... Args> inline std::string Save(std::shared_ptr<TV> const & d, Args const & ... args)
{
	return DataSaver<TV>(d.get(), std::forward<Args const &>(args)...).GetName();
}
template<typename TV, typename ... Args> inline std::string Save(TV* d, Args const & ... args)
{
	return DataSaver<TV>(d, std::forward<Args const &>(args)...).GetName();
}

template<typename TV, typename ... Args> inline std::string Save(std::vector<TV>const & d, std::string const & name,
        Args const & ... args)
{
	size_t s = d.size();
	return DataSaver<TV>(&d[0], name, 1, &s, std::forward<Args const &>(args)...).GetName();
}
template<typename TL, typename TR, typename ... Args> inline std::string Save(std::map<TL, TR>const & d,
        std::string const & name, Args const & ... args)
{
	std::vector<std::pair<TL, TR> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(p);
	}
	return Save(d_, name, std::forward<Args const &>(args)...);
}

template<typename TV, typename ... Args> inline std::string Save(std::map<TV, TV>const & d, std::string const & name,
        Args const & ... args)
{
	std::vector<nTuple<2, TV> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(nTuple<2, TV>( { p.first, p.second }));
	}
	return Save(d_, name, std::forward<Args const &>(args)...);
}
template<typename U>
std::ostream & operator<<(std::ostream & os, DataSaver<U> const &d)
{
	os << d.GetName();
	return os;
}
#define DUMP(_F_) simpla::Save(_F_,__STRING(_F_) ,true)
#define DUMP1(_F_) simpla::Save(_F_,__STRING(_F_) ,false)
#ifndef NDEBUG
#	define DEBUG_DUMP(_F_) simpla::Save(_F_,__STRING(_F_),true)
#else
#   define DEBUG_DUMP(_F_) ""
#endif
}  // namespace simpla

#endif /* DATA_DUMPER_H_ */
