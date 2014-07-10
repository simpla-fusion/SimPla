/*
 * data_stream.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 *
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/log.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/data_type.h"
namespace simpla
{
/** \defgroup  DataIO Data input/output system
 *  @{
 *   \defgroup  HDF5  HDF5 interface
 *   \defgroup  XDMF   XDMF interface
 *   \defgroup  NetCDF  NetCDF interface
 *     \brief UNIMPLEMENT notfix
 *  @}
 *  */

/**
 * \ingroup DataIO
 * @class DataStream
 * \brief data stream , should be a singleton
 */
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
	void Init(int argc = 0, char** argv = nullptr);

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

	bool is_ready() const;

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

	template<typename TV, typename ...Args>
	std::string Write(std::string const & name, TV const *data, Args && ...args) const
	{
		return WriteRawData(name, reinterpret_cast<void const*>(data), DataType::create<TV>(),
		        std::forward<Args>(args)...);
	}

	template<typename TV>
	std::string UnorderedWrite(std::string const & name, TV const *data, size_t number) const
	{
		return WriteUnorderedRawData(name, reinterpret_cast<void const*>(data), DataType::create<TV>(), number);
	}

	template<typename TV>
	std::string UnorderedWrite(std::string const & name, std::vector<TV> const &data) const
	{

		return WriteUnorderedRawData(name, reinterpret_cast<void const*>(&data[0]), DataType::create<TV>(), data.size());
	}
private:

	std::string WriteRawData(std::string const &name, void const *v,

	DataType const & datatype,

	int rank,

	size_t const *global_begin,

	size_t const *global_end,

	size_t const *local_outer_begin,

	size_t const *local_outer_end,

	size_t const *local_inner_begin,

	size_t const *local_inner_end,

	bool is_append = false

	) const;

	std::string WriteUnorderedRawData(std::string const &name, void const *v, DataType const & datatype,
	        size_t number) const;

	struct pimpl_s;
	pimpl_s *pimpl_;

}
;

//! Global data stream entry
#define GLOBAL_DATA_STREAM  SingletonHolder<DataStream> ::instance()

template<typename TV, typename ...Args>
std::string save(std::string const & name, TV const *data, Args && ...args)
{
	return GLOBAL_DATA_STREAM.Write(name, data , std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(std::string const & name,
        std::shared_ptr<TV> const & d, Args && ... args)
{
	return save(name, d.get(), std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(std::string const & name, std::vector<TV>const & d,
        Args && ... args)
{
	size_t s = 0;
	size_t n = d.size();

	return save(name, &d[0], 1, &s, &n, std::forward<Args>(args)...);
}
template<typename TL, typename TR, typename ... Args> inline std::string save(std::string const & name,
        std::map<TL, TR>const & d, Args && ... args)
{
	std::vector<std::pair<TL, TR> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(p);
	}
	return save(name, d_, std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(std::string const & name, std::map<TV, TV>const & d,
        Args && ... args)
{
	std::vector<nTuple<2, TV> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(nTuple<2, TV>( { p.first, p.second }));
	}

	return save(name, d_, std::forward<Args>(args)...);
}

#define SAVE(_F_) simpla::save(__STRING(_F_),_F_  )
#ifndef NDEBUG
#	define DEBUG_SAVE(_F_) simpla::save(__STRING(_F_),_F_ )
#else
#   define DEBUG_SAVE(_F_) ""
#endif
}
// namespace simpla

#endif /* DATA_STREAM_ */
