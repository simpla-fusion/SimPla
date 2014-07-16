/*
 * data_stream.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 *
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utilities/data_type.h"
#include "../utilities/ntuple.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/any.h"

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
 * @todo add write cache
 */
class DataStream
{
public:

	enum
	{
		SP_APPEND = 1UL << 2, SP_CACHE = SP_APPEND | (1UL << 3), SP_RECORD = SP_APPEND | (1UL << 4),

		SP_UNORDER = (1UL << 5)
	};

	DataStream();

	~DataStream();

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, Any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	void init(int argc = 0, char** argv = nullptr);

	std::string cd(std::string const & url, unsigned int flag = 0UL);

	std::string pwd() const;

	void close();

	bool command(std::string const & cmd);

	bool is_ready() const;

	/**
	 *
	 * @param name             dataset name or path
	 * @param v                pointer to data
	 * @param datatype		   data type
	 * @param ndims_or_number  if data shapes are  nullptr , represents the number of dimensions;
	 *                         else represents the  number of data
	 *
	 * \group data shape
	 * \{
	 * @param global_begin
	 * @param global_end
	 * @param local_outer_begin
	 * @param local_outer_end
	 * @param local_inner_begin
	 * @param local_inner_end
	 * \}
	 * @param flag             flag to define the operation
	 * @return
	 */
	std::string write(std::string const &name, void const *v,

	DataType const & datatype,

	size_t ndims_or_number,

	size_t const *global_begin = nullptr,

	size_t const *global_end = nullptr,

	size_t const *local_outer_begin = nullptr,

	size_t const *local_outer_end = nullptr,

	size_t const *local_inner_begin = nullptr,

	size_t const *local_inner_end = nullptr,

	unsigned int flag = 0UL

	);

private:
	struct pimpl_s;

	std::unique_ptr<pimpl_s> pimpl_;

	void set_property_(std::string const & name, Any const&);
	Any get_property_(std::string const & name) const;

}
;

//! Global data stream entry
#define GLOBAL_DATA_STREAM  SingletonHolder<DataStream> ::instance()

template<typename TV, typename ...Args>
std::string save(std::string const & name, TV const *data, Args && ...args)
{
	return GLOBAL_DATA_STREAM.write(name, data , DataType::create<TV>(), std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(std::string const & name,
        std::shared_ptr<TV> const & d, Args && ... args)
{
	return GLOBAL_DATA_STREAM.write(name, d.get(), DataType::create<TV>(), std::forward<Args>(args)...);
}

template<typename TV> inline std::string save(std::string const & name, std::vector<TV>const & d)
{

	size_t s = d.size();
	return GLOBAL_DATA_STREAM.write(name, &d[0], DataType::create<TV>(),1,nullptr,&s );
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
