/**
 * @file io.h
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#ifndef CORE_IO_IO_H_
#define CORE_IO_IO_H_

#include <stddef.h>
#include <string>

#include "../gtl/utilities/utilities.h"
#include "../dataset/dataset.h"
#include "../dataset/datatype.h"
#include "data_stream.h"

namespace simpla
{

struct DataSet;

namespace io
{
/** @addtogroup io
 *  @brief this module collects stuff used to read/write data file.
 * @{
 */


void init(int argc, char **argv);

void close();

std::string help_message();

std::string cd(std::string const &url);


std::string save(std::string const &url, DataSet const &ds, size_t flag = 0UL);

template<typename T>
std::string save(std::string const &url, T const &d, size_t flag = 0UL)
{
    return save(url, traits::make_dataset(d), flag);
}

void delete_attribute(std::string const &url);

void set_dataset_attribute(std::string const &url, std::string const &str);

void set_dataset_attribute(std::string const &url, any const &prop);

any get_dataset_attribute(std::string const &url);

template<typename T> void set_dataset_attribute(std::string const &url, T const &v)
{
    set_dataset_attribute(url, any(v));
}

template<typename T>
T get_dataset_attribute(std::string const &url)
{
    return std::move(get_dataset_attribute(url).template as<T>());
}


//template<typename Tuple, size_t ...Is>
//std::string save_tuple_impl(std::string const & name, Tuple const & d,
//		index_sequence<Is...>)
//{
//	return std::move(save(name, std::get<Is>(d)...));
//}
//
//template<typename ...T>
//std::string save(std::string const & name, std::tuple<T...> const & d,
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
//std::string save(std::string const & name, TV const *data, Args && ...args)
//{
//	return GLOBAL_DATA_STREAM.write(name, data, make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> inline std::string save(
//		std::string const & name, std::shared_ptr<TV> const & d,
//		Args && ... args)
//{
//	return GLOBAL_DATA_STREAM.write(name, d.get(), make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV> std::string save(std::string const & name,
//		std::vector<TV>const & d, size_t flag = 0UL)
//{
//
//	size_t s = d.size();
//	return GLOBAL_DATA_STREAM.write(name, &d[0], make_datatype<TV>(), 1,
//			nullptr, &s, nullptr, nullptr, nullptr, nullptr, flag);
//}
//
//template<typename TL, typename TR, typename ... Args> std::string save(
//		std::string const & name, std::map<TL, TR>const & d, Args && ... args)
//{
//	std::vector<std::pair<TL, TR> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(p);
//	}
//	return save(name, d_, std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> std::string save(
//		std::string const & name, std::map<TV, TV>const & d, Args && ... args)
//{
//	std::vector<nTuple<TV, 2> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(nTuple<TV, 2>(
//		{ p.first, p.second }));
//	}
//
//	return save(name, d_, std::forward<Args>(args)...);
//}
/** @} */

}//namespace io



#define SAVE(_F_) simpla::io::save(__STRING(_F_),_F_  )
#define SAVE_APPEND(_F_) simpla::io::save(__STRING(_F_),_F_, simpla::io::SP_APPEND  )
#define SAVE_RECORD(_F_) simpla::io::save(__STRING(_F_),_F_, simpla::io::SP_RECORD  )

#ifndef NDEBUG
#	define DEBUG_SAVE(_F_) simpla::io::save(__STRING(_F_),_F_ )
#else
#   define DEBUG_SAVE(_F_) ""
#endif
} // namespace simpla

#endif /* CORE_IO_IO_H_ */
