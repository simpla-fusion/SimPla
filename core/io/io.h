/*
 * io.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef CORE_IO_IO_H_
#define CORE_IO_IO_H_

#include <stddef.h>
#include <string>

#include "../utilities/utilities.h"
#include "../data_interface/data_type.h"
namespace simpla
{
struct DataSet;

enum
{
	SP_APPEND = 1UL << 2, SP_CACHE = (1UL << 3), SP_RECORD = (1UL << 4),

	SP_UNORDER = (1UL << 5)
};

void init_io(int argc, char ** argv);
void close_io();
std::string cd(std::string const & url);

std::string save(std::string const & url, DataSet const & ds,
		size_t flag = 0UL);

template<typename T>
auto save(std::string const & name, T const & d, size_t flag = 0UL)
DECL_RET_TYPE(save(name, make_dataset(d), flag))

#define SAVE(_F_) simpla::save(__STRING(_F_),_F_  )
#ifndef NDEBUG
#	define DEBUG_SAVE(_F_) simpla::save(__STRING(_F_),_F_ )
#else
#   define DEBUG_SAVE(_F_) ""
#endif

void delete_attribute(std::string const &url);

void set_dataset_attribute(std::string const &url, std::string const & str);

void set_dataset_attribute(std::string const &url, Any const & prop);

Any get_dataset_attribute(std::string const &url);

template<typename T>
void set_dataset_attribute(std::string const & url, T const&v)
{
	set_dataset_attribute(url, Any(v));
}
template<typename T>
T get_dataset_attribute(std::string const & url)
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

}// namespace simpla

#endif /* CORE_IO_IO_H_ */
