/*
 * sp_integer_sequence.h
 *
 *  Created on: 2014年9月26日
 *      Author: salmon
 */

#ifndef SP_INTEGER_SEQUENCE_H_
#define SP_INTEGER_SEQUENCE_H_

#include <stddef.h>

#include "sp_type_traits.h"

namespace simpla
{
template<typename ...>class _nTuple;
template<typename _Tp, _Tp ... _Idx>
struct integer_sequence
{
private:
	static constexpr size_t size_ = (sizeof...(_Idx));
public:
	typedef _Tp value_type;

	static constexpr _nTuple<size_t, integer_sequence<unsigned int, size_>> value()
	{
		return std::move(_nTuple<size_t, integer_sequence<unsigned int, size_>>(
		{ _Idx... }));
	}
	static constexpr size_t size()
	{
		return size_;
	}

};

template<typename ...> class cat_integer_sequence;

template<typename T, T ... N1, T ... N2>
struct cat_integer_sequence<integer_sequence<T, N1...>,
		integer_sequence<T, N2...>>
{
	typedef integer_sequence<T, N1..., N2...> type;
};

template<unsigned int N, typename ...> struct seq_get_value;

template<unsigned int N, typename Tp, Tp M, Tp ...I>
struct seq_get_value<N, integer_sequence<Tp, M, I ...> >
{
	static constexpr Tp value =
			seq_get_value<N - 1, integer_sequence<Tp, I ...> >::value;
};

template<typename Tp, Tp M, Tp ...I>
struct seq_get_value<0, integer_sequence<Tp, M, I ...> >
{
	static constexpr Tp value = M;
};

template<typename ...> struct seq_for;

template<unsigned int M, unsigned int ...N>
struct seq_for<integer_sequence<unsigned int, M, N...>>
{

	template<typename TOP, typename ...Args>
	static inline void eval_multi_parameter(TOP const & op, Args && ... args)
	{

		seq_for<integer_sequence<unsigned int, N...>>::eval_multi_parameter(op,
				std::forward<Args>(args)..., M - 1);
		seq_for<integer_sequence<unsigned int, M - 1, N...> >::eval_multi_parameter(
				op, std::forward<Args>(args)...);
	}

	template<typename TOP, typename ...Args>
	static inline void eval_ndarray(TOP const & op, Args && ... args)
	{

		seq_for<integer_sequence<unsigned int, N...>>::eval_ndarray(op,
				get_value( std::forward<Args>(args),M - 1)...);
		seq_for<integer_sequence<unsigned int, M - 1, N...> >::eval_ndarray(op,
				std::forward<Args>(args)...);
	}

	template<typename TOP, typename IndexPack, typename ...Args>
	static inline void eval_index_pack(TOP const & op,
			IndexPack const &idx_pack, Args && ... args)
	{

		seq_for<integer_sequence<unsigned int, N...>>::eval_index_pack(op,
				typename cat_integer_sequence<IndexPack,
						integer_sequence<unsigned int, M - 1>>::type(),
				std::forward<Args>(args)...);
		seq_for<integer_sequence<unsigned int, M - 1, N...> >::eval_multi_parameter(
				op, idx_pack, std::forward<Args>(args)...);
	}
};

template<unsigned int ...N>
struct seq_for<integer_sequence<unsigned int, 0, N...>>
{
	template<typename ...Args>
	static inline void eval_multi_parameter(Args && ... args)
	{
	}

	template<typename ...Args>
	static inline void eval_ndarray(Args && ... args)
	{
	}

	template<typename ...Args>
	static inline void eval_index_pack(Args && ... args)
	{
	}
};

template<>
struct seq_for<integer_sequence<unsigned int>>
{
	template<typename TOP, typename ...Args>
	static inline void eval_multi_parameter(TOP const & op, Args && ... args)
	{
		op(std::forward<Args>(args)...);
	}
	template<typename TOP, typename ...Args>
	static inline void eval_ndarray(TOP const & op, Args && ... args)
	{
		op(std::forward<Args>(args)...);
	}

	template<typename TOP, typename IndexPack, typename ...Args>
	static inline void eval_index_pack(TOP const & op,
			IndexPack const & idx_pack, Args && ... args)
	{
		op(get_value(std::forward<Args>(args),idx_pack)...);
	}
};

template<typename ...> struct seq_reduce;

template<unsigned int M, unsigned int ...N>
struct seq_reduce<integer_sequence<unsigned int, M, N...>>
{

	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_multi_parameter(TOP const & op,
			Reduction const & reduction,
			Args && ... args)
					DECL_RET_TYPE(
							(
									reduction(
											seq_reduce<integer_sequence<unsigned int, N...>>::eval_multi_parameter(op,reduction,
													std::forward<Args>(args)..., M - 1),
											seq_reduce<integer_sequence<unsigned int, M - 1, N...> >::eval_multi_parameter(op,reduction,
													std::forward<Args>(args)...)
									)
							)
					)

	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_ndarray(TOP const & op, Reduction const & reduction,
			Args && ... args)
					DECL_RET_TYPE(
							(
									reduction(
											seq_reduce<integer_sequence<unsigned int, N...>>::eval_ndarray(op,reduction,
													get_value(std::forward<Args>(args),M-1)...),
											seq_reduce<integer_sequence<unsigned int, M - 1, N...> >::eval_ndarray(op,reduction,
													std::forward<Args>(args)...)
									)
							)
					)

};

template<unsigned int ...N>
struct seq_reduce<integer_sequence<unsigned int, 1, N...> >
{

	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_multi_parameter(TOP const & op,
			Reduction const & reduction,
			Args && ... args)
					DECL_RET_TYPE(
							( seq_reduce<integer_sequence<unsigned int, N...>>::eval_multi_parameter(op,reduction, std::forward<Args>(args)..., 0) )
					)

	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_ndarray(TOP const & op, Reduction const & reduction,
			Args && ... args)
					DECL_RET_TYPE(
							( seq_reduce<integer_sequence<unsigned int, N...>>::eval_ndarray(op,reduction,get_value( std::forward<Args>(args),0)...) )
					)
};

template<>
struct seq_reduce<integer_sequence<unsigned int> >
{
	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_multi_parameter(TOP const & op,
			Reduction const & reduction, Args && ... args)
			DECL_RET_TYPE ((op(std::forward<Args> (args)...)))

	template<typename TOP, typename Reduction, typename ...Args>
	static inline auto eval_ndarray(TOP const & op, Reduction const & reduction,
			Args && ... args)
			DECL_RET_TYPE ((op(std::forward<Args> (args)...)))

};

}
// namespace simpla
#endif /* SP_INTEGER_SEQUENCE_H_ */
