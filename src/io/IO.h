/**
 * @file IO.h
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#ifndef CORE_IO_IO_H_
#define CORE_IO_IO_H_

#include <stddef.h>
#include <string>

#include "../gtl/utilities/utilities.h"
#include "../data_model/DataSet.h"
#include "../data_model/DataType.h"
#include "HDF5Stream.h"

namespace simpla { namespace io
{
/** @addtogroup io
 *  @brief this module collects stuff used to read/write data file.
 * @{
 */


void init(int argc, char **argv);

void close();

//std::string help_message();

std::string cd(std::string const &url);


std::string write(std::string const &url, data_model::DataSet const &ds, size_t flag = 0UL);

template<typename T>
std::string write(std::string const &url, T const &d, size_t flag = 0UL)
{
    return write(url, data_model::DataSet::create(d), flag);
}


template<typename T>
std::string write(std::string const &url, size_t num, T const *d, size_t flag = 0UL)
{
    return write(url, data_model::DataSet::create(d, num), flag);
}
//
//void delete_attribute(std::string const &url);
//
//void set_data_set_attribute(std::string const &url, std::string const &str);
//
//void set_data_set_attribute(std::string const &url, any const &prop);
//
//any get_data_set_attribute(std::string const &url);
//
//template<typename T> void set_data_set_attribute(std::string const &url, T const &v)
//{
//    set_data_set_attribute(url, any(v));
//}
//
//template<typename T>
//T get_data_set_attribute(std::string const &url)
//{
//    return std::move(get_data_set_attribute(url).template as<T>());
//}
//

//template<typename Tuple, size_t ...Is>
//std::string save_tuple_impl(std::string const & name, Tuple const & d,
//		index_sequence<Is...>)
//{
//	return std::move(write(name, std::get<Is>(d)...));
//}
//
//template<typename ...T>
//std::string write(std::string const & name, std::tuple<T...> const & d,
//		size_t flag = 0UL)
//{
//	return std::move(save_tuple_impl(name, d,
//
//	make_index_sequence<sizeof...(T)>()
//
//	));
//}
//
//template<typename TV, typename ...Args>
//std::string write(std::string const & name, TV const *data, Args && ...args)
//{
//	return GLOBAL_DATA_STREAM.write(name, data, make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> inline std::string write(
//		std::string const & name, std::shared_ptr<TV> const & d,
//		Args && ... args)
//{
//	return GLOBAL_DATA_STREAM.write(name, d.get(), make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV> std::string write(std::string const & name,
//		std::vector<TV>const & d, size_t flag = 0UL)
//{
//
//	size_t s = d.size();
//	return GLOBAL_DATA_STREAM.write(name, &d[0], make_datatype<TV>(), 1,
//			nullptr, &s, nullptr, nullptr, nullptr, nullptr, flag);
//}
//
//template<typename TL, typename TR, typename ... Args> std::string write(
//		std::string const & name, std::map<TL, TR>const & d, Args && ... args)
//{
//	std::vector<std::pair<TL, TR> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(p);
//	}
//	return write(name, d_, std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> std::string write(
//		std::string const & name, std::map<TV, TV>const & d, Args && ... args)
//{
//	std::vector<nTuple<TV, 2> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(nTuple<TV, 2>(
//		{ p.first, p.second }));
//	}
//
//	return write(name, d_, std::forward<Args>(args)...);
//}
/** @} */

}} // namespace simpla //namespace io

namespace simpla
{


#define SAVE(_F_) simpla::io::write(__STRING(_F_),_F_  )
#define SAVE_APPEND(_F_) simpla::io::write(__STRING(_F_),_F_, simpla::io::SP_APPEND  )
#define SAVE_RECORD(_F_) simpla::io::write(__STRING(_F_),_F_, simpla::io::SP_RECORD  )

#ifndef NDEBUG
#	define DEBUG_SAVE(_F_) simpla::io::write(__STRING(_F_),_F_ )
#else
#   define DEBUG_SAVE(_F_) ""
#endif
}//namespace simpla{

#endif /* CORE_IO_IO_H_ */
