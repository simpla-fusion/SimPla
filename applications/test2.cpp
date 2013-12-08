/*
 * test2.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <iostream>
#include "../src/utilities/log.h"

template<typename ...TI>
void Foo(TI ...s)
{
	std::cout << sizeof...(s) << std::endl;
}
template<typename TFUN,typename
int main(int argc, char **argv)
{
	long s = 16 | 32 | 2;

	CHECK(((((s) & 3) + 1) % 3 - 1));
	CHECK(((((s >> 2) & 3) + 1) % 3 - 1));
	CHECK(((((s >> 4) & 3) + 1) % 3 - 1));
}
enum
{
	WITH_GHOSTS = 1, NO_GHOSTS = 0
};

void Traversal(int IFORM,
		std::function<void(size_t, size_t, size_t, size_t)> const &fun,
		int flag = NO_GHOSTS) const
{
	size_t ib = (flag == NO_GHOSTS) ? 0 : gw_[0];
	size_t ie = (flag == NO_GHOSTS) ? dims_[0] : dims_[0] - gw_[0];

	size_t jb = (flag == NO_GHOSTS) ? 0 : gw_[1];
	size_t je = (flag == NO_GHOSTS) ? dims_[1] : dims_[1] - gw_[1];

	size_t kb = (flag == NO_GHOSTS) ? 0 : gw_[2];
	size_t ke = (flag == NO_GHOSTS) ? dims_[2] : dims_[2] - gw_[2];

	size_t mb = 0;
	size_t me = num_comps_per_cell_[IFORM];

	for (size_t i = ib; i < ie; ++i)
		for (size_t j = jb; j < je; ++j)
			for (size_t k = kb; k < ke; ++k)
				for (size_t m = mb; m < me; ++m)
				{
					fun(m, i, j, k);
				}

}

//	void Traversal2(int IFORM, std::function<void(int, size_t)> const &fun,
//			int flag = NO_GHOSTS) const
//	{
//		Traversal(IFORM,
//
//		[](size_t m,size_t i,size_t j ,size_t k)
//		{
//			fun(m,GetIndex(i,j,k));
//		},
//
//		flag);
//	}

// For All

template<typename Fun, typename TF, typename ...Args> inline
void ForAll(Fun const &fun, TF * l, Args const & ... args) const
{
	Traversal(FieldTraits<TF>::IForm,

	[&](size_t m,size_t i,size_t j,size_t k)
	{
		fun(
				get(l,m,i,j,k),

				this_type::template get(args,m,i,j,k)...);
	}, WITH_GHOSTS);

}
template<typename Fun, typename TF, typename ...Args> inline
void ForAll(Fun const &fun, TF const & l, Args const & ... args) const
{
	Traversal(FieldTraits<TF>::IForm,
			[&](size_t m,size_t i,size_t j,size_t k)
			{	fun(get(l,m,i,j,k), get(args,m,i,j,k)...);},
			WITH_GHOSTS);

}
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForAll(Fun const &fun, TF const & l, Args const& ... args) const
//	{
//
//		int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];
//
//		for (size_t i = 0; i < dims_[0]; ++i)
//			for (size_t j = 0; j < dims_[1]; ++j)
//				for (size_t k = 0; k < dims_[2]; ++k)
//					for (size_t m = 0; m < num_comp; ++m)
//					{
//						fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
//					}
//	}
//
//	template<typename Fun, typename TF, typename ...Args> inline
//	void ForAll(Fun const &fun, TF * l, Args const & ... args) const
//	{
//		int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];
//
//		for (size_t i = 0; i < dims_[0]; ++i)
//			for (size_t j = 0; j < dims_[1]; ++j)
//				for (size_t k = 0; k < dims_[2]; ++k)
//					for (size_t m = 0; m < num_comp; ++m)
//					{
//						fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
//					}
//	}

template<typename Fun, typename TF, typename ... Args> inline
void ForEach(Fun const &fun, TF const & l, Args const& ... args) const
{

	int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];

	for (size_t i = gw_[0]; i < dims_[0] - gw_[0]; ++i)
		for (size_t j = gw_[1]; j < dims_[1] - gw_[1]; ++j)
			for (size_t k = gw_[2]; k < dims_[2] - gw_[2]; ++k)
				for (size_t m = 0; m < num_comp; ++m)
				{
					fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
				}
}

template<typename Fun, typename TF, typename ...Args> inline
void ForEach(Fun const &fun, TF * l, Args const & ... args) const
{
	int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];

	for (size_t i = gw_[0]; i < dims_[0] - gw_[0]; ++i)
		for (size_t j = gw_[1]; j < dims_[1] - gw_[1]; ++j)
			for (size_t k = gw_[2]; k < dims_[2] - gw_[2]; ++k)
				for (size_t m = 0; m < num_comp; ++m)
				{
					fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
				}
}

//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForEach(Fun const &fun, TF * l, Args const& ... args) const
//	{
//		Traversal(FieldTraits<TF>::IForm, [&](size_t m,size_t i,size_t j,size_t k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, NO_GHOSTS);
//	}
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForEach(Fun const &fun, TF const & l, Args const& ... args) const
//	{
//
//		Traversal(FieldTraits<TF>::IForm, [&](size_t m,size_t i,size_t j,size_t k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, NO_GHOSTS);
//	}
