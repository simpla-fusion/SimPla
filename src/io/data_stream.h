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

#include <complex>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "../fetl/ntuple.h"
#include "../utilities/data_type.h"
#include "../utilities/log.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/pretty_stream.h"

namespace simpla
{

class DataStream
{
	std::string prefix_;
	int suffix_width_;

	std::string filename_;
	std::string grpname_;
	size_t LIGHT_DATA_LIMIT_;
	bool enable_compact_storable_;
	bool enable_xdmf_;

public:

	DataStream();
	~DataStream();
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
		enable_compact_storable_ = true;
	}
	void DisableCompactStorable()
	{
		enable_compact_storable_ = false;
	}

	void EnableXDMF()
	{
		enable_xdmf_ = true;
	}
	void DisableXDMF()
	{
		enable_xdmf_ = false;
	}

	bool CheckCompactStorable() const
	{
		return enable_compact_storable_;
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

	void OpenGroup(std::string const & gname);
	void OpenFile(std::string const &fname = "unnamed");
	void CloseGroup();
	void CloseFile();

	void Close()
	{
		CloseGroup();
		CloseFile();
	}

	template<typename TV, typename ... Args>
	std::string Write(std::string const &name, TV const *v, Args ...args) const
	{
		return WriteHDF5(name, reinterpret_cast<void const*>(v), DataType<TV>().Desc(),
				std::forward<Args const & >(args)...);
	}

	std::string WriteHDF5(std::string const &name, void const *v, DataTypeDesc const & mdtype,

	int rank,

	size_t const *global_dims,

	size_t const *local_outer_start = nullptr,

	size_t const *local_outer_count = nullptr,

	size_t const *local_inner_start = nullptr,

	size_t const *local_inner_count = nullptr) const;

private:

	struct pimpl_s;
	pimpl_s *pimpl_;

}
;

#define GLOBAL_DATA_STREAM  SingletonHolder<DataStream> ::instance()

template<typename TV, typename ...Args>
inline std::string Save(std::string const & name, TV const *data, Args const & ...args)
{
	return GLOBAL_DATA_STREAM.Write(name,data, std::forward<Args const &>(args)...);
}

template<typename TV, typename ... Args> inline std::string Save(std::string const name, std::shared_ptr<TV> const & d,
		Args const & ... args)
{
	return Save(name, d.get(), std::forward<Args const &>(args)...);
}

template<typename TV, int rank, typename TS> inline std::string Save(std::string const &name, TV const* data,
		nTuple<rank, TS> const & d)
{
	return Save(name, reinterpret_cast<void const *>(data), rank, &d[0]);
}

template<typename TV, typename ... Args> inline std::string Save(std::string const & name, std::vector<TV>const & d,
		Args const & ... args)
{
	size_t s = d.size();

	return Save(name, &d[0], 1, &s, std::forward<Args const &>(args)...);
}
template<typename TL, typename TR, typename ... Args> inline std::string Save(std::string const & name,
		std::map<TL, TR>const & d, Args const & ... args)
{
	std::vector<std::pair<TL, TR> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(p);
	}
	return Save(name, d_, std::forward<Args const &>(args)...);
}

template<typename TV, typename ... Args> inline std::string Save(std::string const & name, std::map<TV, TV>const & d,
		Args const & ... args)
{
	std::vector<nTuple<2, TV> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(nTuple<2, TV>(
		{ p.first, p.second }));
	}
	return Save(name, d_, std::forward<Args const &>(args)...);
}

#define SAVE(_F_) simpla::Save(__STRING(_F_),_F_  )
#ifndef NDEBUG
#	define DEBUG_SAVE(_F_) simpla::Save(__STRING(_F_),_F_ )
#else
#   define DEBUG_SAVE(_F_) ""
#endif
}
// namespace simpla

#endif /* DATA_STREAM_ */
