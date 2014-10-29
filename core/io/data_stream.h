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
 */
class DataStream
{
public:

	enum
	{
		SP_APPEND = 1UL << 2, SP_CACHE = (1UL << 3), SP_RECORD = (1UL << 4),

		SP_UNORDER = (1UL << 5)
	};

	DataStream();

	~DataStream();

	Properties & properties();
	Properties const & properties() const;

	template<typename T> void properties(std::string const & name, T const&v)
	{
		properties().set(name, Any(v));
	}

	template<typename T> T properties(std::string const & name) const
	{
		return properties().get(name).template as<T>();
	}

	void init(int argc = 0, char** argv = nullptr);

	std::string cd(std::string const & url, size_t is_append = 0UL);

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
	std::string write(std::string const &name,

	void const *v,

	DataType const & datatype,

	size_t ndims_or_number,

	size_t const *global_begin = nullptr,

	size_t const *global_end = nullptr,

	size_t const *local_outer_begin = nullptr,

	size_t const *local_outer_end = nullptr,

	size_t const *local_inner_begin = nullptr,

	size_t const *local_inner_end = nullptr,

	size_t flag = 0UL

	);

	/**
	 *
	 * @param url  <file name>:/<group path>/<obj name>.<attribute>
	 * @param d_type
	 * @param v
	 */

	void set_attribute(std::string const &url, DataType const & d_type,
			void const * buff);

	void get_attribute(std::string const &url, DataType const & d_type,
			void* buff);

	void delete_attribute(std::string const &url);

	void set_attribute(std::string const &url, char const str[])
	{
		set_attribute(url, std::string(str));
	}

	template<typename T> void set_attribute(std::string const & url, T const&v)
	{
		set_attribute(url, make_datatype<T>(), &v);
	}
	template<typename T>
	T get_attribute(std::string const & url)
	{
		T res;

		get_attribute(url, make_datatype<T>(), &res);

		return std::move(res);
	}

private:
	struct pimpl_s;

	std::unique_ptr<pimpl_s> pimpl_;

}
;

//! Global data stream entry
#define GLOBAL_DATA_STREAM  SingletonHolder<DataStream> ::instance()

template<typename Tuple, size_t ...Is>
std::string save_tuple_impl(std::string const & name, Tuple const & d,
		index_sequence<Is...>)
{
	return std::move(GLOBAL_DATA_STREAM.write(name, std::get<Is>(d)... ));
}
template<typename ...T>
std::string save(std::string const & name, std::tuple<T...> const & d,
		size_t flag = 0UL)
{
	return std::move(save_tuple_impl(name, d,

	make_index_sequence<sizeof...(T)>()

	));
}

template<typename TV, typename ...Args>
std::string save(std::string const & name, TV const *data, Args && ...args)
{
	return GLOBAL_DATA_STREAM.write(name, data , make_datatype<TV>(), std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(
		std::string const & name, std::shared_ptr<TV> const & d,
		Args && ... args)
{
	return GLOBAL_DATA_STREAM.write(name, d.get(), make_datatype<TV>(), std::forward<Args>(args)...);
}

template<typename TV> inline std::string save(std::string const & name,
		std::vector<TV>const & d, size_t flag = 0UL)
{

	size_t s = d.size();
	return GLOBAL_DATA_STREAM.write(name, &d[0], make_datatype<TV>(),1,nullptr,&s,
			nullptr,nullptr,nullptr,nullptr ,flag );
}

template<typename TL, typename TR, typename ... Args> inline std::string save(
		std::string const & name, std::map<TL, TR>const & d, Args && ... args)
{
	std::vector<std::pair<TL, TR> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(p);
	}
	return save(name, d_, std::forward<Args>(args)...);
}

template<typename TV, typename ... Args> inline std::string save(
		std::string const & name, std::map<TV, TV>const & d, Args && ... args)
{
	std::vector<nTuple<TV, 2> > d_;
	for (auto const & p : d)
	{
		d_.emplace_back(nTuple<TV, 2>(
		{ p.first, p.second }));
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
