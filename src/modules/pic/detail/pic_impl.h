/*
 * pic_impl.h
 *
 *  Created on: 2012-2-14
 *      Author: salmon
 */

#ifndef PIC_IMPL_H_
#define PIC_IMPL_H_
/** @file detail/pic_impl.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{pic.h}
 */
#include <omp.h>

namespace simpla
{

namespace pic
{
template<typename TP, typename TG>
void PIC<TP, TG>::AllocMemory(size_t num_of_ele)
{
	num_of_ele_ = num_of_ele;
	element_size_in_bytes_ = EngineType::get_element_size_in_bytes();

	size_t memory_size = num_of_ele_ * element_size_in_bytes_;

	try
	{
		TR1::shared_ptr<ByteType>(
				reinterpret_cast<ByteType*>(operator new(memory_size))).swap(
				data_);
	} catch (std::bad_alloc const & error)
	{
		ERROR_BAD_ALLOC_MEMORY(memory_size, error);
	}

	try
	{
		StorageType::resize(grid.get_num_of_cell());
	} catch (std::bad_alloc const & error)
	{
		ERROR_BAD_ALLOC_MEMORY(grid.get_num_of_cell() * sizeof(size_t), error);
	}
}
template<typename TP, typename TG>
void PIC<TP, TG>::Initialize(ZeroForm const &pn1, size_t pic)
{
	AllocMemory(pic * grid.get_num_of_cell());

	RandomLoad();

	size_t se = grid.get_num_of_cell();

	IVec3 w =
	{ 1, 1, 1 };

#pragma omp parallel for
	for (size_t s = 0; s < se; ++s)
	{

		StorageType::operator[](s) = get_particle(s * pic);

		RVec3 X0 = grid.get_cell_center(s);

		Grid sgrid = grid.SubGrid(X0, w);

		ZeroForm n1(sgrid);

		n1 = pn1;

		for (int i = 0; i < pic; ++i)
		{
			Point_s * p = get_particle(s * pic + i);

			EngineType::PreProcess(n1, X0, sgrid.dx, 1.0, p);

			p->next = get_particle(s * pic + i + 1);

		}
		get_particle(s * pic + pic - 1)->next = NULL;

	}

}

template<typename TP, typename TG>
template<typename TContext>
void PIC<TP, TG>::PushAndScatter(TContext &ctx)
{

	Interpolation<VecZeroForm> B0(ctx.find("B0")->second);
	Interpolation<OneForm> E1(ctx.find("E1")->second);
	Interpolation<TwoForm> B1(ctx.find("B1")->second);

	Interpolation<VecZeroForm> J1(ctx.find("J1")->second);
	Interpolation<ZeroForm> n1(ctx.find("n1")->second);

#pragma omp parallel for
	for (size_t s = 0; s < StorageType::size(); ++s)
	{
		Grid subgrid = grid.SubGrid();
		B0.Prefetch(s);
		B1.Prefetch(s);
		E1.Prefetch(s);

		J1.Cache(s);
		n1.Cache(s);

		for (Point_s * np = StorageType::operator[](s); np != NULL;
				np = np->next)
		{
			EngineType::PushAndScatter(B0, E1, B1, J1, n1, grid.dt, np);
		}

	}
}

} // namespace pic
} // namespace simpla

#endif /* PIC_IMPL_H_ */
