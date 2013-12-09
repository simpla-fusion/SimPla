/*
 * serialize.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef SERIALIZE_H_
#define SERIALIZE_H_

#include <type_traits>

#include "../fetl/primitives.h"
#include "../mesh/mesh.h"

namespace simpla
{
namespace _impl
{

template<typename T, typename TConfig>
class has_serialize_mem_fn
{

	template<typename T1, typename T2>
	static auto check_op(T1 const& obj, T2 * cfg)
	->decltype(obj.template Serialize<T2>(cfg) )
	{
	}

	static std::false_type check_op(...)
	{
		return std::false_type();
	}

	typedef decltype( check_op(std::declval<T>(), std::declval<TConfig*>()) ) result_type;

public:

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

};

template<typename T, typename TConfig>
class has_deserialize_mem_fn
{

	template<typename T1, typename T2>
	static auto check_op(T1 & obj, T2 const & cfg)
	->decltype(obj.template Deserialize<T2>(cfg) )
	{
	}

	static std::false_type check_op(...)
	{
		return std::false_type();
	}

	typedef decltype( check_op(std::declval<T>(), std::declval<TConfig>()) ) result_type;

public:

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

};

}  // namespace _impl

template<typename T, typename TConfig>
auto Serialize(T const& obj, TConfig *cfg)
ENABLE_IF_DECL_RET_TYPE(
		( _impl::has_serialize_mem_fn<T , TConfig>::value),
		(obj.Serialize(cfg)))

template<typename T, typename TConfig>
auto Serialize(T const& obj, TConfig *cfg)
ENABLE_IF_DECL_RET_TYPE(
		(!_impl::has_serialize_mem_fn<T , TConfig>::value),
		(_Serialize(obj,cfg)))

template<typename T, typename TConfig>
auto Deserialize(T const& obj, TConfig *cfg)
ENABLE_IF_DECL_RET_TYPE(
		( _impl::has_serialize_mem_fn<T , TConfig>::value),
		(obj.Deserialize(cfg)))

template<typename T, typename TConfig>
auto Deserialize(T const& obj, TConfig *cfg)
ENABLE_IF_DECL_RET_TYPE(
		(!_impl::has_serialize_mem_fn<T , TConfig>::value),
		(_Deserialize(obj,cfg)))

}
// namespace fetl_impl

}  // namespace simpla

#endif /* SERIALIZE_H_ */
