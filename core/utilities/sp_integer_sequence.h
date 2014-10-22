/*
 * sp_integer_sequence.h
 *
 *  Created on: 2014年9月26日
 *      Author: salmon
 */

#ifndef SP_INTEGER_SEQUENCE_H_
#define SP_INTEGER_SEQUENCE_H_

#include <stddef.h>
#include <tuple>
#include "sp_type_traits.h"

namespace simpla
{

template<typename _Tp, _Tp ... _Idx>
struct integer_sequence
{
private:
	static constexpr size_t size_ = (sizeof...(_Idx));
public:
	typedef _Tp value_type;

	static constexpr size_t size() noexcept
	{
		return size_;
	}

};

template<typename T, typename TInts>
T& get_value(T &v, integer_sequence<TInts>)
{
	return v;
}

template<typename T, typename TInts, TInts M, TInts ... N>
auto get_value(T & v, integer_sequence<TInts, M, N...>)
DECL_RET_TYPE((get_value(v[M],integer_sequence<TInts , N...>()) ))
template<typename T, typename TInts, TInts M, TInts ... N>
auto get_value(T* v, integer_sequence<TInts, M, N...>)
DECL_RET_TYPE((get_value(v[M],integer_sequence<TInts , N...>()) ))

template<size_t ... N> using index_sequence = integer_sequence<size_t , N...>;

template<size_t N, typename ...> struct seq_get_value;

template<size_t N, typename Tp, Tp M, Tp ...I>
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

template<typename ...> class cat_integer_sequence;

template<typename T, T ... N1, T ... N2>
struct cat_integer_sequence<integer_sequence<T, N1...>,
		integer_sequence<T, N2...>>
{
	typedef integer_sequence<T, N1..., N2...> type;
};

template<size_t...> struct _seq_for;

template<size_t M, size_t ...N>
struct _seq_for<M, N...>
{

	template<typename TOP, typename ...Args>
	static inline void eval(TOP const & op, Args & ... args)
	{
		eval(op, integer_sequence<size_t>(), (args)...);
	}

	template<typename TOP, size_t ...L, typename ...Args>
	static inline void eval(TOP const & op, integer_sequence<size_t, L...>,
			Args && ... args)
	{
		_seq_for<N...>::eval(op, integer_sequence<size_t, L..., M>(),
				std::forward<Args>(args)...);

		_seq_for<M - 1, N...>::eval(op, integer_sequence<size_t, L...>(),
				std::forward<Args>(args)...);
	}

};

template<size_t ...N>
struct _seq_for<0, N...>
{
	template<typename ...Args>
	static inline void eval(Args && ... args)
	{
	}
};

template<>
struct _seq_for<>
{

	template<typename TOP, typename TInts, TInts ...L, typename ...Args>
	static inline void eval(TOP const & op, integer_sequence<TInts, L...>,
			Args && ... args)
	{
		typedef integer_sequence<size_t, (L-1)...> i_seq;

		op(get_value(std::forward<Args>(args),i_seq()) ...);

	}
};

template<size_t ... N, typename ...Args>
void seq_for(integer_sequence<size_t, N...>, Args && ... args)
{
	_seq_for<N...>::eval(std::forward<Args>(args) ...);
}

template<size_t...> struct _seq_reduce;

template<size_t M, size_t ...N>
struct _seq_reduce<M, N...>
{

	template<typename Reduction, size_t ...L, typename ... Args>
	static inline auto eval(Reduction const & reduction,
			integer_sequence<size_t, L...>,
			Args const&... args)
					DECL_RET_TYPE(
							(
									reduction( _seq_reduce< N... >::eval( reduction,
													integer_sequence<size_t , L..., M>(),
													std::forward<Args const>(args)... ),

											_seq_reduce< M - 1, N... >::eval( reduction,
													integer_sequence<size_t , L...>(),
													std::forward<Args const> (args)... ) )
							))

	template<typename Reduction, typename ...Args>
	static inline auto eval(Reduction const & reduction,
			Args const& ... args)
					DECL_RET_TYPE(
							( eval( reduction,integer_sequence<size_t >(), std::forward<Args const> (args)...) ))

};

template<size_t ...N>
struct _seq_reduce<1, N...>
{

	template<typename Reduction, size_t ...L, typename ...Args>
	static inline auto eval(Reduction const & reduction,
			integer_sequence<size_t, L...>,
			Args const &... args)
					DECL_RET_TYPE(
							(
									_seq_reduce< N... >::eval(
											reduction, integer_sequence<size_t , L..., 1>(),
											std::forward<Args const>(args)... )
							)
					)

};

template<>
struct _seq_reduce<>
{
	template<typename Reduction, size_t ...L, typename Args>
	static inline auto eval(Reduction const &, integer_sequence<size_t, L...>,
			Args const & args)
			DECL_RET_TYPE( (get_value( (args),
									integer_sequence<size_t , (L-1)...> ()) ))

};
template<size_t ... N, typename ...Args>
auto seq_reduce(integer_sequence<size_t, N...>, Args && ... args)
DECL_RET_TYPE(std::move (_seq_reduce<N...>::eval(std::forward<Args>(args) ...)))

template<typename TOS, size_t...> struct _seq_print;

template<typename TOS, size_t M, size_t ...N>
struct _seq_print<TOS, M, N...>
{

	template<size_t ...L, typename T>
	static inline void eval(TOS & os, integer_sequence<size_t, L...>,
			T const & args)
	{
		_seq_print<TOS, N...>::eval(os, integer_sequence<size_t, L..., M>(),
				args);

		_seq_print<TOS, M - 1, N...>::eval(os, integer_sequence<size_t, L...>(),
				args);
	}

};

template<typename TOS, size_t ...N>
struct _seq_print<TOS, 0, N...>
{
	template<typename ... T>
	static inline void eval(TOS & os, T &&...)
	{
		os << std::endl;
	}
};

template<typename TOS>
struct _seq_print<TOS>
{

	template<typename T, size_t ...N>
	static inline void eval(TOS & os, integer_sequence<size_t, N...>,
			T const& args)
	{
		typedef integer_sequence<size_t, (N-1)...> i_seq;
		os << (get_value(args, i_seq())) << ",";
	}

};
}
// namespace simpla
#endif /* SP_INTEGER_SEQUENCE_H_ */
