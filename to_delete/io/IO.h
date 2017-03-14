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


#include "IOStream.h"
#include "HDF5Stream.h"

namespace simpla { namespace toolbox
{

/**
 * @ingroup toolbox
 * @addtogroup io
 * @brief this module collects stuff used to read/write m_data file.
 * @{
 */


std::shared_ptr<IOStream> create_from_output_url(std::string const &url);
//
//void close();
//
//IOStream &global();
//
////std::string help_message();
//
//std::string cd(std::string const &url);
//
//
//std::string write(std::string const &url, data_model::DataSet const &ds, size_t id = 0UL);
//
//template<typename T>
//std::string write(std::string const &url, T const &d, size_t id = 0UL)
//{
//    return write(url, data_model::DataSet::clone(d), id);
//}
//
//
//template<typename T>
//std::string write(std::string const &url, size_t num, T const *d, size_t id = 0UL)
//{
//    return write(url, data_model::DataSet::clone(d, num), id);
//}
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
//    return std::Move(get_data_set_attribute(url).template as<T>());
//}
//

//template<typename Tuple, size_t ...Is>
//std::string save_tuple_impl(std::string const & GetName, Tuple const & d,
//		int_sequence<Is...>)
//{
//	return std::Move(write(GetName, std::Get<Is>(d)...));
//}
//
//template<typename ...T>
//std::string write(std::string const & GetName, std::tuple<T...> const & d,
//		size_t id = 0UL)
//{
//	return std::Move(save_tuple_impl(GetName, d,
//
//	make_int_sequence<sizeof...(T)>()
//
//	));
//}
//
//template<typename TV, typename ...Args>
//std::string write(std::string const & GetName, TV const *m_data, Args && ...args)
//{
//	return GLOBAL_DATA_STREAM.write(GetName, m_data, make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> inline std::string write(
//		std::string const & GetName, std::shared_ptr<TV> const & d,
//		Args && ... args)
//{
//	return GLOBAL_DATA_STREAM.write(GetName, d.Get(), make_datatype<TV>(),
//			std::forward<Args>(args)...);
//}
//
//template<typename TV> std::string write(std::string const & GetName,
//		std::vector<TV>const & d, size_t id = 0UL)
//{
//
//	size_t s = d.size();
//	return GLOBAL_DATA_STREAM.write(GetName, &d[0], make_datatype<TV>(), 1,
//			nullptr, &s, nullptr, nullptr, nullptr, nullptr, id);
//}
//
//template<typename TL, typename TR, typename ... Args> std::string write(
//		std::string const & GetName, std::map<TL, TR>const & d, Args && ... args)
//{
//	std::vector<std::pair<TL, TR> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(p);
//	}
//	return write(GetName, d_, std::forward<Args>(args)...);
//}
//
//template<typename TV, typename ... Args> std::string write(
//		std::string const & GetName, std::map<TV, TV>const & d, Args && ... args)
//{
//	std::vector<nTuple<TV, 2> > d_;
//	for (auto const & p : d)
//	{
//		d_.emplace_back(nTuple<TV, 2>(
//		{ p.first, p.second }));
//	}
//
//	return write(GetName, d_, std::forward<Args>(args)...);
//}
/** @} */

}} // namespace simpla //namespace io
//
//namespace simpla
//{
//#define  SAVE_(_OS_, _F_, __FLAG__)                                                         \
//{        auto pwd = _OS_.pwd();                                                           \
//        _OS_.open( std::string("/")+__STRING(_F_)+"/");                                                         \
//        _OS_.write(_F_.mesh()->GetName(), _F_.dataset(mesh_as::SP_ES_ALL), __FLAG__);            \
//        _OS_.open(pwd);                                                                     \
//}
//#define SAVE(_OS_, _F_) SAVE_(_OS_,_F_,::simpla::io::SP_NEW)
//#define SAVE_APPEND(_F_) SAVE_(_OS_,_F_,::simpla::io::SP_APPEND  )
//#define SAVE_RECORD(_F_) SAVE_(_OS_,_F_,::simpla::io::SP_RECORD  )
//
//#ifndef NDEBUG
//#	define DEBUG_SAVE(_F_) simpla::io::write(__STRING(_F_),_F_ )
//#else
//#   define DEBUG_SAVE(_F_) ""
//#endif
//}//namespace simpla{

#endif /* CORE_IO_IO_H_ */
