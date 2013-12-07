/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

#include <fetl/field.h>
#include <fetl/geometry.h>
#include <fetl/ntuple.h>
#include <fetl/primitives.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <set>
#include "utilities/log.h"
#include "physics/physical_constants.h"
#include "mesh.h"
namespace simpla
{

/**
 *
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh 
 * */

template<typename TS = Real>
struct CoRectMesh
{
	typedef CoRectMesh this_type;

	static const int NUM_OF_DIMS = 3;

	typedef size_t index_type;

	typedef TS scalar;

	typedef nTuple<3, Real> coordinates_type;

	typedef unsigned int tag_type;

	PhysicalConstants phys_constants;

	this_type & operator=(const this_type&) = delete;

	Real dt_ = 0.0;

	enum
	{
		DEFAULT_GHOST_WIDTH = 2
	};
	// Topology
	nTuple<NUM_OF_DIMS, size_t> shift_;

	nTuple<NUM_OF_DIMS, size_t> dims_ /*={ 11, 11, 11 }*/;

	nTuple<NUM_OF_DIMS, size_t> gw_ =
	{ DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH };

	nTuple<NUM_OF_DIMS, size_t> strides_;

	nTuple<NUM_OF_DIMS, size_t> period_;

	static const int num_vertices_in_cell_ = 8;

	size_t num_cells_ = 0;
	size_t num_grid_points_ = 0;

	// Geometry
	coordinates_type xmin_ =
	{ 0, 0, 0 };
	coordinates_type xmax_ =
	{ 10, 10, 10 };
	nTuple<NUM_OF_DIMS, scalar> dS_[2];
	nTuple<NUM_OF_DIMS, scalar> dx_;

	Real cell_volume_ = 1.0;
	Real d_cell_volume_ = 1.0;

	const size_t num_comps_per_cell_[4] =
	{ 1, 3, 3, 1 };

	size_t idx_shift_[8];

	std::vector<tag_type> media_tag_;
	std::vector<index_type> element_in_boundary_[4];
	std::vector<index_type> element_on_boundary_[4];

	CoRectMesh()
	{
	}

	~CoRectMesh()
	{
	}

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	template<typename PT>
	inline void Deserialize(PT const &vm)
	{
		phys_constants.Deserialize(vm.GetChild("UnitSystem"));

		vm.GetChild("Topology").template GetValue("Dimensions", &dims_);
		vm.GetChild("Topology").template GetValue("GhostWidth", &gw_);
		vm.GetChild("Geometry").template GetValue("Min", &xmin_);
		vm.GetChild("Geometry").template GetValue("Max", &xmax_);

		Update();
	}

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		vm.GetChild("Topology").template SetValue("Dimensions", &dims_);
		vm.GetChild("Topology").template SetValue("GhostWidth", &gw_);
		vm.GetChild("Geometry").template SetValue("Min", &xmin_);
		vm.GetChild("Geometry").template SetValue("Max", &xmax_);

		phys_constants.Serialize(vm.GetChild("UnitSystem"));
	}

	inline void _SetImaginaryPart(Real i, Real * v)
	{
	}

	inline void _SetImaginaryPart(Real i, Complex * v)
	{
		v->imag(i);
	}
	void Update()
	{
		num_cells_ = 1;
		num_grid_points_ = 1;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			gw_[i] = (gw_[i] * 2 > dims_[i]) ? dims_[i] / 2 : gw_[i];
			if (dims_[i] <= 1)
			{
				dims_[i] = 1;
				dx_[i] = 0.0;

				dS_[0][i] = 0.0;
				_SetImaginaryPart(
						xmax_[i] == xmin_[i] ? 0 : 1.0 / (xmax_[i] - xmin_[i]),
						&dS_[0][i]);
				dS_[1][i] = 0.0;
			}
			else
			{
				dx_[i] = (xmax_[i] - xmin_[i])
						/ static_cast<Real>(dims_[i] - 1);
				dS_[0][i] = 1.0 / dx_[i];
				dS_[1][i] = -1.0 / dx_[i];

				num_cells_ *= (dims_[i] - 1);
				num_grid_points_ *= dims_[i];
			}
		}
		strides_[2] = 1;
		strides_[1] = dims_[2];
		strides_[0] = dims_[1] * dims_[2];

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (dims_[i] == 1)
			{
				strides_[i] = 0;
			}

			if (!isinf(dx_[i]))
			{
				cell_volume_ *= (dims_[i] - 1) * dx_[i];
			}

			if (!isinf(dx_[i]) && dx_[i] > 0)
			{
				d_cell_volume_ *= dx_[i];
			}

			UnSetPeriodicBoundary(i);
		}
		media_tag_.resize(num_grid_points_, 0);

	}

	template<int iform, typename E> inline typename Container<E>::type MakeContainer() const
	{
		return std::move(Container<E>::Create(GetNumOfGridPoints(iform)));
	}

	/**
	 *
	 * @param fun if fun(x)==true , vertex(x) is set to meida (media_idx)
	 * @param media_idx > 0
	 */
	template<typename TFUN>
	void AssignDomainMedia(TFUN const &fun, tag_type media_tag)
	{

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					size_t s = (i * strides_[0] + j * strides_[1]
							+ k * strides_[2]);

					coordinates_type x = xmin_;
					x[0] += dx_[0] * i;
					x[1] += dx_[0] * j;
					x[2] += dx_[0] * k;

					if (fun(x) == true)
					{
						media_tag_[s] = media_tag;
					}
				}

	}
	enum
	{
		PARALLEL, PERPENDICULAR
	};

	template<int IFORM, int DIRECTION>
	void GetElementOnInterface(tag_type in, tag_type out,
			std::vector<index_type>*res)
	{
		std::set<index_type> tmp_res;

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					size_t s0 = (i * strides_[0] + j * strides_[1]
							+ k * strides_[2]);

					tag_type v[8];

					/** @ref The Visualization Toolkit 4th Ed. p.273
					 *
					 *                ^y
					 *               /
					 *        z     /
					 *        ^
					 *        |   6---------------7
					 *        |  /|              /|
					 *          / |             / |
					 *         /  |            /  |
					 *        4---|-----------5   |
					 *        | --> B0        |   |
					 *        |   2-----------|---3
					 *        E2 /    ^B2       |  /
					 *        | E1    |     | /
					 *        |/              |/
					 *        0------E0-------1   ---> x
					 *
					 *
					 */

					v[0] = media_tag_[s0];
					v[1] = media_tag_[s0 + strides_[0]];
					v[2] = media_tag_[s0 + strides_[1]];
					v[3] = media_tag_[s0 + strides_[0] + strides_[1]];

					v[4] = media_tag_[s0 + strides_[2]];
					v[5] = media_tag_[s0 + strides_[2] + strides_[0]];
					v[6] = media_tag_[s0 + strides_[2] + strides_[1]];
					v[7] = media_tag_[s0 + strides_[2] + strides_[0]
							+ strides_[1]];

					// not interface
					if (((v[0] = v[1]) && (v[1] = v[2]) && (v[2] = v[3])
							&& (v[3] = v[4]) && (v[4] = v[5]) && (v[5] = v[6])
							&& (v[6] = v[7]))

							|| ((v[0] != in) && (v[1] != in) && (v[2] != in)
									&& (v[3] != in) && (v[4] != in)
									&& (v[5] != in) && (v[6] != in)
									&& (v[7] != in))

							|| ((v[0] != out) && (v[1] != out) && (v[2] != out)
									&& (v[3] != out) && (v[4] != out)
									&& (v[5] != out) && (v[6] != out)
									&& (v[7] != out)))
						continue;

					_SetInterface(Int2Type<IFORM>(), Int2Type<DIRECTION>(), s0,
							in, v, &tmp_res);

				}

		res->clear();
		std::copy(tmp_res.begin(), tmp_res.end(), std::back_inserter(*res));

	}

private:

	template<int DIRECTION>
	void _SetInterface(Int2Type<0>, Int2Type<DIRECTION>, index_type s0,
			tag_type in, tag_type const* v, std::set<index_type> *res)
	{
		if (v[0] == in)
			res->insert((s0));
	}
	void _SetInterface(Int2Type<1>, Int2Type<PARALLEL>, index_type s0,
			tag_type in, tag_type const* v, std::set<index_type> *res)
	{
		if ((v[0] == in) && (v[0] == v[1]))
			res->insert((s0) * 3 + 0);

		if ((v[2] == v[3]) && (v[3] == in))
			res->insert((s0 + strides_[1]) * 3 + 0);

		if ((v[4] == v[5]) && (v[4] == in))
			res->insert((s0 + strides_[2]) * 3 + 0);

		if ((v[6] == v[7]) && (v[7] == in))
			res->insert((s0 + strides_[2] + strides_[1]) * 3 + 0);

		//

		if ((v[0] == v[2]) && (v[2] == in))
			res->insert((s0) * 3 + 1);

		if ((v[1] == v[3]) && (v[1] == in))
			res->insert((s0 + strides_[0]) * 3 + 1);

		if ((v[4] == v[6]) && (v[6] == in))
			res->insert((s0 + strides_[2]) * 3 + 1);

		if ((v[5] == v[7]) && (v[5] == in))
			res->insert((s0 + strides_[2] + strides_[0]) * 3 + 1);

		//

		if ((v[0] == v[4]) && (v[0] == in))
			res->insert((s0) * 3 + 2);

		if ((v[1] == v[5]) && (v[1] == in))
			res->insert((s0 + strides_[0]) * 3 + 2);

		if ((v[2] == v[6]) && (v[2] == in))
			res->insert((s0 + strides_[1]) * 3 + 2);

		if ((v[3] == v[7]) && (v[3] == in))
			res->insert((s0 + strides_[0] + strides_[1]) * 3 + 2);

	}
	void _SetInterface(Int2Type<1>, Int2Type<PERPENDICULAR>, index_type s0,
			tag_type in, tag_type const* v, std::set<index_type> *res)
	{
		if ((v[0] != v[1]))
			res->insert((s0) * 3 + 0);

		if ((v[2] != v[3]))
			res->insert((s0 + strides_[1]) * 3 + 0);

		if ((v[4] != v[5]))
			res->insert((s0 + strides_[2]) * 3 + 0);

		if ((v[6] != v[7]))
			res->insert((s0 + strides_[2] + strides_[1]) * 3 + 0);

		//

		if ((v[0] != v[2]))
			res->insert((s0) * 3 + 1);

		if ((v[1] != v[3]))
			res->insert((s0 + strides_[0]) * 3 + 1);

		if ((v[4] != v[6]))
			res->insert((s0 + strides_[2]) * 3 + 1);

		if ((v[5] != v[7]))
			res->insert((s0 + strides_[2] + strides_[0]) * 3 + 1);

		//

		if ((v[0] != v[4]))
			res->insert((s0) * 3 + 2);

		if ((v[1] != v[5]))
			res->insert((s0 + strides_[0]) * 3 + 2);

		if ((v[2] != v[6]))
			res->insert((s0 + strides_[1]) * 3 + 2);

		if ((v[3] != v[7]))
			res->insert((s0 + strides_[0] + strides_[1]) * 3 + 2);

	}

	void _SetInterface(Int2Type<2>, Int2Type<PARALLEL>, index_type s0,
			tag_type in, tag_type const* v, std::set<index_type> *res)
	{

		if (!((v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3])))
			res->insert((s0) * 3 + 2);
		if (!((v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7])))
			res->insert((s0 + strides_[2]) * 3 + 2);

		if (!((v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5])))
			res->insert((s0) * 3 + 1);
		if (!((v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7])))
			res->insert((s0 + strides_[1]) * 3 + 1);

		if (!((v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6])))
			res->insert((s0) * 3 + 0);
		if (!((v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7])))
			res->insert((s0 + strides_[0]) * 3 + 1);

	}

	void _SetInterface(Int2Type<2>, Int2Type<PERPENDICULAR>, index_type s0,
			tag_type in, tag_type const* v, std::set<index_type> *res)
	{

		if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3]))
			res->insert((s0) * 3 + 2);
		if ((v[4] == in) && (v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7]))
			res->insert((s0 + strides_[2]) * 3 + 2);

		if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5]))
			res->insert((s0) * 3 + 1);
		if ((v[2] == in) && (v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7]))
			res->insert((s0 + strides_[1]) * 3 + 1);

		if ((v[0] == in) && (v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6]))
			res->insert((s0) * 3 + 0);
		if ((v[1] == in) && (v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7]))
			res->insert((s0 + strides_[0]) * 3 + 1);

	}

	void _SetInterface(Int2Type<3>, index_type s0, tag_type in,
			tag_type const* v, std::set<index_type> *res)
	{

		if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3]))
			res->insert((s0 - strides_[2]));
		if ((v[4] == in) && (v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7]))
			res->insert((s0 + strides_[2]));

		if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5]))
			res->insert((s0 - strides_[1]));
		if ((v[2] == in) && (v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7]))
			res->insert((s0 + strides_[1]));

		if ((v[0] == in) && (v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6]))
			res->insert((s0 + strides_[0]));
		if ((v[1] == in) && (v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7]))
			res->insert((s0 + strides_[0]));

		WARNING
				<< "This implement is incorrect when the boundary has too sharp corner";

		/**
		 *  FIXME this is incorrect when the boundary has too sharp corner
		 *  for example
		 *                  ^    out
		 *                 / \
		 *       @--------/-@-\----------@
		 *       |       /  |  \         |
		 *       |      /   |   \        |
		 *       |     /    |    \       |
		 *       |    /     |     \      |
		 *       @---/------@------\-----@
		 *          /               \
		 *         /       in        \
		 */

	}
public:
//
//	template<int IFORM, typename T1>
//	void Print(Field<Geometry<this_type, IFORM>, T1> const & f) const
//	{
//		size_t num_comp = num_comps_per_cell_[IFORM];
//
//		for (size_t i = 0; i < dims_[0]; ++i)
//		{
//			std::cout << "--------------------------------------------------"
//					<< std::endl;
//			for (size_t j = 0; j < dims_[1]; ++j)
//			{
//				std::cout << std::endl;
//				for (size_t k = 0; k < dims_[2]; ++k)
//				{
//					std::cout << "(";
//					for (size_t m = 0; m < num_comp; ++m)
//					{
//						std::cout
//								<< f[(i * strides_[0] + j * strides_[1]
//										+ k * strides_[2]) * num_comp + m]
//								<< " ";
//					}
//					std::cout << ") ";
//				}
//				std::cout << std::endl;
//			}
//
//		}
//		std::cout << std::endl;
//
//	}

	template<typename Fun, typename ... Args> inline
	void ForAllCell(Fun const &fun, Args &... args) const
	{

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					size_t s = (i * strides_[0] + j * strides_[1]
							+ k * strides_[2]);

					fun(s, args...);
				}

	}
	template<typename Fun, typename ... Args> inline
	void ForAllVertex(Fun const &fun, Args &... args) const
	{

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					size_t s = (i * strides_[0] + j * strides_[1]
							+ k * strides_[2]);

					coordinates_type x = xmin_;
					x[0] += dx_[0] * i;
					x[1] += dx_[0] * j;
					x[2] += dx_[0] * k;
					fun(s, x, args...);
				}

	}

	inline size_t GetIndex(index_type i, index_type j, index_type k) const
	{
		return ((i % period_[0]) * strides_[0] + (j % period_[1]) * strides_[1]
				+ (k % period_[2]) * strides_[2]);
	}
	inline size_t GetIndex(index_type s) const
	{
		return s;
	}
	template<int IFORM, typename ... IDXS>
	inline size_t Component(int m, IDXS ... s) const
	{
		return GetIndex(s...) * num_comps_per_cell_[IFORM] + m;
	}

private:

	enum
	{
		NIL = 0, // 00 00 00
		X = 1,   // 00 00 01
		NX = 2,   // 00 00 10
		Y = 4,   // 00 01 00
		NY = 8,   // 00 10 00
		Z = 16,  // 01 00 00
		NZ = 32   // 10 00 00
	};
	inline size_t INC(int m) const
	{
		return 1 << (m % 3) * 2;
	}
	inline size_t DES(int m) const
	{
		return 2 << (m % 3) * 2;
	}

public:

	/**
	 * (((d & 3) + 1) % 3 - 1)
	 *
	 * 00 -> 0
	 * 01 -> 1
	 * 10 -> -1
	 *
	 */
	template<typename ... IDXS>
	inline size_t Shift(int d, IDXS ... s) const
	{

		return GetIndex(s...)

		+ ((((d >> 4) & 3) + 1) % 3 - 1) * strides_[2]

		+ ((((d >> 2) & 3) + 1) % 3 - 1) * strides_[1]

		+ (((d & 3) + 1) % 3 - 1) * strides_[0];

		;
	}

	inline size_t Shift(int d, index_type i, index_type j, index_type k) const
	{
		return

		(((i + (((d & 3) + 1) % 3 - 1)) % period_[0]) * strides_[0]

		+ ((j + ((((d >> 2) & 3) + 1) % 3 - 1)) % period_[1]) * strides_[1]

		+ ((k + ((((d >> 4) & 3) + 1) % 3 - 1)) % period_[2]) * strides_[2]);
	}
	template<typename T, typename ... TI>
	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l,
			TI ...) const
	{
		return std::move(l);
	}

	template<int IFORM, typename TL, typename ...TI> inline typename Field<
			Geometry<this_type, IFORM>, TL>::value_type & get(
			Field<Geometry<this_type, IFORM>, TL> *l, TI ... s) const
	{
		return l->get(s...);
	}

	template<int IFORM, typename TL, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, TL>::value_type const &get(
			Field<Geometry<this_type, IFORM>, TL> const & l, TI ...s) const
	{
		return (l.get(s...));
	}

	template<int IFORM, int TOP, typename TL, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> >::value_type get(
			Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> > const & l,
			TI ...s) const
	{
		return (l.get(s...));
	}

	template<int IFORM, int TOP, typename TL, typename TR, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> >::value_type get(
			Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> > const & l,
			TI ...s) const
	{
		return (l.get(s...));
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ForAll(Fun const &fun, TF const & l, Args const& ... args) const
	{

		int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_comp; ++m)
					{
						fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
					}
	}

	template<typename Fun, typename TF, typename ...Args> inline
	void ForAll(Fun const &fun, TF * l, Args const & ... args) const
	{
		int num_comp = num_comps_per_cell_[FieldTraits<TF>::IForm];

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_comp; ++m)
					{
						fun(get(l, m, i, j, k), get(args, m,i, j, k)...);
					}
	}

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

// Properties of UniformRectMesh --------------------------------------
	inline void SetPeriodicBoundary(int i)
	{
		period_[i] = dims_[i];
	}
	inline void UnSetPeriodicBoundary(int i)
	{
		period_[i] = std::numeric_limits<index_type>::max();
	}

	inline void SetExtent(coordinates_type const & pmin,
			coordinates_type const & pmax)
	{
		xmin_ = pmin;
		xmax_ = pmax;

		Update();
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline void SetDimension(nTuple<NUM_OF_DIMS, size_t> const & pdims)
	{
		dims_ = pdims;

		Update();
	}
	inline nTuple<NUM_OF_DIMS, size_t> const & GetDimension() const
	{
		return dims_;
	}
	inline nTuple<NUM_OF_DIMS, size_t> const & GetStrides() const
	{
		return strides_;
	}

// General Property -----------------------------------------------

	inline Real GetDt() const
	{
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}

	inline size_t GetNumOfGridPoints(int iform) const
	{

		return (num_grid_points_ * num_comps_per_cell_[iform]);
	}

	inline size_t GetNumOfVertex() const
	{

		return (num_grid_points_);
	}

	inline Real GetCellVolume() const
	{
		return cell_volume_;
	}

	inline Real GetDCellVolume() const
	{
		return d_cell_volume_;
	}

	/**
	 * Locate the cell containing a specified point.
	 * @param x
	 * @param pcoords local parameter coordinates
	 * @return index of cell
	 */
	inline index_type SearchCell(coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const
	{

		size_t idx = 0;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			double e;

			idx +=
					static_cast<size_t>(std::modf((x[i] - xmin_[i]) * dS_[i],
							&e)) * strides_[i];

			if (pcoords != nullptr)
				(*pcoords)[i] = e;
		}

		return idx;
	}
	/**
	 *  Speed up version SearchCell
	 * @param
	 * @param x
	 * @param pcoords
	 * @return
	 */
	inline index_type SearchCell(index_type const &hint_idx,
			coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const
	{
		return SearchCell(x, pcoords);
	}

	inline std::vector<coordinates_type> GetCellShape(index_type s) const
	{
		std::vector<coordinates_type> res;
		coordinates_type r0 =
		{ 0, 0, 0 };
		res.push_back(GetGlobalCoordinates(s, r0));
		coordinates_type r1 =
		{ 1, 1, 1 };
		res.push_back(GetGlobalCoordinates(s, r1));

		return res;
	}

	inline coordinates_type GetGlobalCoordinates(index_type s,
			coordinates_type const &r) const
	{
		coordinates_type res;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (strides_[i] != 0)
			{
				res[i] = (s / strides_[i]) * dx_[i] + xmin_[i];

				s %= strides_[i];
			}
			else
			{
				res[i] = xmin_[i];
			}
		}
		res += r * dx_;

		return res;

	}
	template<typename PList>
	inline void GetAffectedPoints(Int2Type<0>, index_type const & idx,
			PList & points, int affect_region = 1) const
	{

		points.resize(8);
		// 0 0 0
		points[0] = idx;
		// 0 1 0
		points[1] = idx + strides_[0];
		// 0 0 1
		points[2] = idx + strides_[1];
		// 0 1 1
		points[3] = idx + strides_[0] + strides_[1];
		// 1 0 0
		points[4] = idx + strides_[2];
		// 1 1 0
		points[5] = idx + strides_[0] + strides_[2];
		// 1 0 1
		points[6] = idx + strides_[1] + strides_[2];
		// 1 1 1
		points[7] = idx + strides_[0] + strides_[1] + strides_[2];

	}

	template<typename PList>
	inline void GetAffectedPoints(Int2Type<1>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}

	template<typename PList>
	inline void GetAffectedPoints(Int2Type<2>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}
	template<typename PList>
	inline void GetAffectedPoints(Int2Type<3>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<0>, coordinates_type const &pcoords,
			TW & weights, int affect_region = 1) const
	{
		weights.resize(8);
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];

		weights[0] = (1.0 - r) * (1.0 - s) * (1.0 - t);
		weights[1] = r * (1.0 - s) * (1.0 - t);
		weights[2] = (1.0 - r) * s * (1.0 - t);
		weights[3] = r * s * (1.0 - t);
		weights[4] = (1.0 - r) * (1.0 - s) * t;
		weights[5] = r * (1.0 - s) * t;
		weights[6] = (1.0 - r) * s * t;
		weights[7] = r * s * t;
	}
	template<typename TW>
	inline void CalcuateWeights(Int2Type<1>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<2>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<3>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

// Mapto ----------------------------------------------------------
	/**
	 *    mapto(Int2Type<0> ,   //tarGet topology position
	 *     Field<this_type,1 , TExpr> const & vl,  //field
	 *      SizeType s   //grid index of point
	 *      )
	 * target topology position:
	 *             z 	y 	x
	 *       000 : 0,	0,	0   vertex
	 *       001 : 0,	0,1/2   edge
	 *       010 : 0, 1/2,	0
	 *       100 : 1/2,	0,	0
	 *       011 : 0, 1/2,1/2   Face
	 * */

	template<int IF, typename T, typename ...TI> inline double //
	mapto(Int2Type<IF>, T const &l, TI ... s) const
	{
		return (l);
	}

	template<int IF, typename TL, typename ...TI> inline auto mapto(
			Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l,
			TI ... s) const
			DECL_RET_TYPE ((get(l,s...)))

	template<typename TL, typename ...IDXS> inline auto mapto(Int2Type<1>,
			Field<Geometry<this_type, 0>, TL> const &l, index_type m,
			IDXS ... s) const
					DECL_RET_TYPE( ((get(l,m,Shift(INC(m),s...)) + get(l,m,s...))*0.5) )

	template<typename TL, typename ...IDXS> inline auto //
	mapto(Int2Type<2>, Field<Geometry<this_type, 0>, TL> const &l, index_type m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(
											get(l,0,s...)+
											get(l,0,Shift(INC(m+1),s...))+
											get(l,0,Shift(INC(m+2),s...))+
											get(l,0,Shift(INC(m+1) | INC(m+2) ,s...))
									)*0.25

							))
	template<typename TL, typename ...IDXS> inline auto //
	mapto(Int2Type<3>, Field<Geometry<this_type, 0>, TL> const &l, index_type m,
			IDXS ...s) const
			DECL_RET_TYPE(( (
									get(l,0,s...)+
									get(l,0,Shift(X,s...))+
									get(l,0,Shift(Y,s...))+
									get(l,0,Shift(Z,s...))+

									get(l,0,Shift(X|Y,s...))+
									get(l,0,Shift(Z|X,s...))+
									get(l,0,Shift(Z|Y,s...))+
									get(l,0,Shift(Z|X|Y,s...))

							)*0.125

					))

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<0>, Field<Geometry<this_type, 2>, TL> const &l,
			TI ... s) const
			DECL_RET_TYPE( (get(l,s...)) )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<1>, Field<Geometry<this_type, 2>, TL> const &l,
			TI ...s) const
			DECL_RET_TYPE( (get(l,s...)) )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<3>, Field<Geometry<this_type, 2>, TL> const &l,
			TI ... s) const
			DECL_RET_TYPE( (get(l,s...)) )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<0>, Field<Geometry<this_type, 1>, TL> const &l,
			TI ...s) const
			DECL_RET_TYPE( (get(l,s...)) )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<2>, Field<Geometry<this_type, 1>, TL> const &l,
			TI ...s) const
			DECL_RET_TYPE( (get(l,s...)) )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<3>, Field<Geometry<this_type, 1>, TL> const &l,
			TI ... s) const
			DECL_RET_TYPE( (get(l,s...)) )

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

	template<typename TExpr, typename ... IDXS> inline auto OpEval(
			Int2Type<GRAD>, Field<Geometry<this_type, 0>, TExpr> const & f,
			size_t m, IDXS ... s) const
			DECL_RET_TYPE(
					( get(f,0,Shift(INC(m),s...))* dS_[0][m] )
					+get(f,0,s...)* dS_[1][m])

	template<typename TExpr, typename ...IDX> inline auto OpEval(
			Int2Type<DIVERGE>, Field<Geometry<this_type, 1>, TExpr> const & f,
			size_t m,
			IDX ...s) const
					DECL_RET_TYPE((

									(get(f,0,s...)* dS_[0][0] + get(f,0,Shift( NX,s...))* dS_[1][0]) +

									(get(f,1,s...) * dS_[0][1] + get(f,1,Shift( NY,s...))* dS_[1][1]) +

									(get(f,2,s...) * dS_[0][2] + get(f,2,Shift( NZ,s...))* dS_[1][2])
							))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
			Field<Geometry<this_type, 1>, TL> const & f, size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									get(f,(m+2)%3,Shift(INC(m+1) ,s...)) * dS_[0][(m + 1) % 3]

									+ get(f,(m+2)%3,s...)* dS_[1][(m + 1) % 3]

									- get(f,(m+1)%3,Shift(INC(m+2) ,s...)) * dS_[0][(m + 2) % 3]

									- get(f,(m+1)%3,s...)* dS_[1][(m + 2) % 3]
							)
					)

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
			Field<Geometry<this_type, 2>, TL> const & f, size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									get(f,(m+2)%3,s...)* dS_[0][(m + 1) % 3]

									+ get(f,(m+2)%3,Shift(DES(m+1),s...)) * dS_[1][(m + 1) % 3]

									- get(f,(m+1)%3,s...)* dS_[0][(m + 2) % 3]

									- get(f,(m+1)%3,Shift(DES(m+2),s...)) * dS_[1][(m + 2) % 3]

							))

	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDX>, Field<Geometry<this_type, 1>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==0?0:(m==1?2:1)),Shift(X,s...)) * dS_[0][0]
											+ get(f,(m==0?0:(m==1?2:1)),s...)* dS_[1][0])*(m==0?0:(m==1?-1:1))
							))
	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDY>, Field<Geometry<this_type, 1>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==1?0:(m==2?0:2)),Shift(Y,s...)) * dS_[0][1]
											+ get(f,(m==1?0:(m==2?0:2)),s...)* dS_[1][1])*(m==1?0:(m==2?-1:1))
							))

	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDZ>, Field<Geometry<this_type, 1>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==2?0:(m==0?1:0)),Shift(Z,s...)) * dS_[0][2]
											+ get(f,(m==2?0:(m==0?1:0)),s...)* dS_[1][2])*(m==2?0:(m==0?-1:1))
							))

	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDX>, Field<Geometry<this_type, 2>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==0?0:(m==1?2:1)),s...) * dS_[0][0]
											+ get(f,(m==0?0:(m==1?2:1)),Shift(NX,s...))* dS_[1][0])*(m==0?0:(m==1?-1:1))
							))
	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDY>, Field<Geometry<this_type, 2>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==1?0:(m==2?0:2)),s...) * dS_[0][1]
											+ get(f,(m==1?0:(m==2?0:2)),Shift(NY,s...))* dS_[1][1])*(m==1?0:(m==2?-1:1))
							))

	template<typename TL, typename ...IDXS> inline auto OpEval(
			Int2Type<CURLPDZ>, Field<Geometry<this_type, 2>, TL> const & f,
			size_t m,
			IDXS ...s) const
					DECL_RET_TYPE((
									(get(f,(m==2?0:(m==0?1:0)),s...) * dS_[0][2]
											+ get(f,(m==2?0:(m==0?1:0)),Shift(NZ,s...))
											* dS_[1][2])*(m==2?0:(m==0?-1:1))
							))

	template<int N, typename TL, typename ... IDXS> inline auto OpEval(
			Int2Type<EXTRIORDERIVATIVE>,
			Field<Geometry<this_type, N>, TL> const & f, int m, IDXS ... s)
			DECL_RET_TYPE((get(f,m,s...)*dS_[m]))

	template<int IL, int IR, typename TL, typename TR, typename ...TI> inline auto OpEval(
			Int2Type<WEDGE>, Field<Geometry<this_type, IL>, TL> const &l,
			Field<Geometry<this_type, IR>, TR> const &r, TI ... s) const
			DECL_RET_TYPE( ( mapto(Int2Type<IL+IR>(),l,s...)*
							mapto(Int2Type<IL+IR>(),r,s...)))

	template<int IL, typename TL, typename ...TI> inline auto OpEval(
			Int2Type<HODGESTAR>, Field<Geometry<this_type, IL>, TL> const & f,
			TI ... s) const
					DECL_RET_TYPE(( mapto(Int2Type<this_type::NUM_OF_DIMS-IL >(),f,s...)))
}
;

//template<typename TL> inline auto _FieldOpEval(
//		Int2Type<CURLPDY>, Field<Geometry<this_type, 1>, TL> const & f,
//		size_t const & s)
//				DECL_RET_TYPE( (f[s - s %3 + (s + 2) % 3 + 3 * strides_[(s + 1) % 3]]* dS_[0][(s + 1) % 3]
//
//								+ f[s - s %3 + (s + 2) % 3]* dS_[1][(s + 1) % 3]) *((s+1)%3==1?0:1)
//
//						- (f[s - s %3 + (s + 1) % 3 + 3 * strides_[(s + 2) % 3]]* dS_[0][(s + 2) % 3] + f[s - s %3 + (s + 1) % 3]* dS_[1][(s + 2) % 3]) *((s+2)%3==1?0:1)
//
//				)
//
//template<typename TL> inline auto _FieldOpEval(
//		Int2Type<CURLPDZ>, Field<Geometry<this_type, 1>, TL> const & f,
//		size_t const & s)
//				DECL_RET_TYPE( (f[s - s %3 + (s + 2) % 3 + 3 * strides_[(s + 1) % 3]]* dS_[0][(s + 1) % 3] + f[s - s %3 + (s + 2) % 3]* dS_[1][(s + 1) % 3]) *((s+1)%3==2?0:1)
//
//						- (f[s - s %3 + (s + 1) % 3 + 3 * strides_[(s + 2) % 3]] * dS_[0][(s + 2) % 3] + f[s - s %3 + (s + 1) % 3]* dS_[(s + 2) % 3]) *((s+2)%3==2?0:1) )
//
//template<typename TL> inline auto _FieldOpEval(
//		Int2Type<CURLPDX>, Field<Geometry<this_type, 2>, TL> const & f,
//		size_t const & s)
//				DECL_RET_TYPE( (f[s - s % 3 + (s + 2) % 3]* dS_[0][(s + 1) % 3] + f[s - s % 3 + (s + 2) % 3 - 3 * strides_[(s + 1) % 3]]* dS_[1][(s + 1) % 3] ) *((s+1)%3==0?0:1)
//
//						-(f[s - s % 3 + (s + 1) % 3]* dS_[0][(s + 2) % 3] + f[s - s % 3 + (s + 1) % 3 - 3 * strides_[(s + 2) % 3]]* dS_[1][(s + 2) % 3]) *((s+2)%3==0?0:1) )
//
//template<typename TL> inline auto _FieldOpEval(
//		Int2Type<CURLPDY>, Field<Geometry<this_type, 2>, TL> const & f,
//		size_t const & s)
//				DECL_RET_TYPE( (f[s - s % 3 + (s + 2) % 3]* dS_[0][(s + 1) % 3] + f[s - s % 3 + (s + 2) % 3 - 3 * strides_[(s + 1) % 3]]* dS_[1][(s + 1) % 3] ) *((s+1)%3==1?0:1)
//
//						-(f[s - s % 3 + (s + 1) % 3] * dS_[0][(s + 2) % 3] + f[s - s % 3 + (s + 1) % 3 - 3 * strides_[(s + 2) % 3]]* dS_[1][(s + 2) % 3]) *((s+2)%3==2?0:1)
//
//				)
//
//template<typename TL> inline auto _FieldOpEval(
//		Int2Type<CURLPDZ>, Field<Geometry<this_type, 2>, TL> const & f,
//		size_t const & s)
//				DECL_RET_TYPE( (f[s - s % 3 + (s + 2) % 3] * dS_[0][(s + 1) % 3] + f[s - s % 3 + (s + 2) % 3 - 3 * strides_[(s + 1) % 3]]* dS_[1][(s + 1) % 3] ) *((s+1)%3==1?0:1)
//
//						-(f[s - s % 3 + (s + 1) % 3] * dS_[0][(s + 2) % 3] + f[s - s % 3 + (s + 1) % 3 - 3 * strides_[(s + 2) % 3]]* dS_[1][(s + 2) % 3]) *((s+2)%3==2?0:1) )

}
// namespace simpla

///**
// *  Boundary
// * */
//
//
//template<typename Fun> inline
//void ForEachBoundary(int iform, Fun const &f) const
//{
//	size_t num_comp = num_comps_per_cell_[iform];
//
//	for (size_t i = 0; i < dims_[0]; ++i)
//		for (size_t j = 0; j < dims_[1]; ++j)
//			for (size_t k = 0; k < dims_[2]; ++k)
//				for (size_t m = 0; m < num_comp; ++m)
//				{
//					if (i >= gw_[0] && i < dims_[0] - gw_[0] &&
//
//					j >= gw_[1] && j < dims_[1] - gw_[1] &&
//
//					k >= gw_[2] && k < dims_[2] - gw_[2]
//
//					)
//					{
//						continue;
//					}
//					else
//					{
//						f(
//								(i * strides_[0] + j * strides_[1]
//										+ k * strides_[2]) * num_comp + m);
//					}
//
//				}
//
//}
//
//void MakeCycleMap(int iform, std::map<index_type, index_type> &ma,
//		unsigned int flag = 7) const
//{
//	size_t num_comp = num_comps_per_cell_[iform];
//
//	nTuple<NUM_OF_DIMS, size_t> L =
//	{ dims_[0] - 2 * gw_[0], dims_[1] - 2 * gw_[1], dims_[2] - 2 * gw_[2] };
//
//	for (size_t i = 0; i < dims_[0]; ++i)
//		for (size_t j = 0; j < dims_[1]; ++j)
//			for (size_t k = 0; k < dims_[2]; ++k)
//			{
//
//				index_type s = i * strides_[0] + j * strides_[1]
//						+ k * strides_[2];
//				index_type t = s;
//
//				if (flag & 1)
//				{
//					if (i < gw_[0])
//					{
//						t += L[0] * strides_[0];
//					}
//					else if (i >= dims_[0] - gw_[0])
//					{
//						t -= L[0] * strides_[0];
//					}
//				}
//
//				if (flag & 2)
//				{
//					if (j < gw_[1])
//					{
//						t += L[1] * strides_[1];
//					}
//					else if (j >= dims_[1] - gw_[1])
//					{
//						t -= L[1] * strides_[1];
//					}
//				}
//
//				if (flag & 4)
//				{
//					if (k < gw_[2])
//					{
//						t += L[2] * strides_[2];
//					}
//					else if (k >= dims_[2] - gw_[2])
//					{
//						t -= L[2] * strides_[2];
//					}
//				}
//				if (s != t)
//				{
//					for (size_t m = 0; m < num_comp; ++m)
//					{
//						ma[s * num_comp + m] = t * num_comp + m;
//					}
//				}
//
//			}
//}
//
//template<int IFORM, typename T1, typename T2>
//void UpdateBoundary(std::map<index_type, index_type> const & m,
//		Field<Geometry<this_type, IFORM>, T1> & src,
//		Field<Geometry<this_type, IFORM>, T2> & dest) const
//{
//	for (auto & p : m)
//	{
//		dest[p.first] = src[p.second];
//	}
//
//}
//
//template<int IFORM, typename T1>
//void UpdateCyCleBoundary(Field<Geometry<this_type, IFORM>, T1> & f) const
//{
//	std::map<index_type, index_type> m;
//	MakeCycleMap(IFORM, m);
//	UpdateBoundary(m, f, f);
//}
/**
 *
 *
 // Interpolation ----------------------------------------------------------

 template<typename TExpr>
 inline typename Field<Geometry<this_type, 0>, TExpr>::Value //
 Gather(Field<Geometry<this_type, 0>, TExpr> const &f, RVec3 x) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 return (f[s] * (1.0 - r[0]) + f[s + strides_[0]] * r[0]); //FIXME Only for 1-dim
 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 0>, TExpr> & f, RVec3 x,
 typename Field<Geometry<this_type, 0>, TExpr>::Value const v) const
 {
 typename Field<Geometry<this_type, 0>, TExpr>::Value res;
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f.Add(s, v * (1.0 - r[0]));
 f.Add(s + strides_[0], v * r[0]); //FIXME Only for 1-dim

 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 1>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (0.5 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (0.5 + r[0]));
 res[1] = (f[(s) * 3 + 1] * (0.5 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (0.5 + r[1]));
 res[2] = (f[(s) * 3 + 2] * (0.5 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (0.5 + r[2]));
 return res;
 }
 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 1>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
 f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
 f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 2>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (1.0 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (r[0]));
 res[1] = (f[(s) * 3 + 1] * (1.0 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (r[1]));
 res[2] = (f[(s) * 3 + 2] * (1.0 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (r[2]));
 return res;

 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 2>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (1.0 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (r[0]);
 f[(s) * 3 + 1] += v[1] * (1.0 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (r[1]);
 f[(s) * 3 + 2] += v[2] * (1.0 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (r[2]);

 }
 *
 *
 * */

#endif //UNIFORM_RECT_H_
