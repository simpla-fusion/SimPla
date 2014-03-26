/*
 * select.h
 *
 *  Created on: 2014年2月19日
 *      Author: salmon
 */

#ifndef SELECT_H_
#define SELECT_H_

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
namespace simpla
{

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        std::function<bool(typename TM::index_type)> const & select)
{
	for (auto s : mesh.GetRange(IFORM))
	{
		if (select(s))
			op(s);
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        std::function<bool(typename TM::coordinates_type)> const & select)
{
	for (auto s : mesh.GetRange(IFORM))
	{
		if (select(mesh.GetCoordinates(s)))
			op(s);
	}
}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        typename TM::coordinates_type const & x)
{
//	typename TM::index_type idxs[TM::MAX_NUM_NEIGHBOUR_ELEMENT];
//	int n = mesh.template GetAdjacentCells(Int2Type<VOLUME>(), Int2Type<IFORM>(), mesh.GetIndex(x), idxs);
//
//	for (int i = 0; i < n; ++i)
//	{
//		op(idxs[i]);
//	}

	typename TM::index_type v[3];
	int n = mesh.template GetCellIndex<IFORM>(x, 0, v);
	for (int i = 0; i < n; ++i)
	{
		op(v[i]);
	}
}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        typename TM::coordinates_type const & v0, typename TM::coordinates_type const & v1)
{
	for (auto s : mesh.GetRange(IFORM))
	{
		auto x = mesh.GetCoordinates(s);

		if ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
		        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0))
		{
			op(s);
		}
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        std::vector<typename TM::index_type> const & idxs)
{
	typename TM::index_type id[TM::MAX_NUM_NEIGHBOUR_ELEMENT];

	for (auto const & s : idxs)
	{

		int n = mesh.GetAdjacentCells(Int2Type<VOLUME>(), Int2Type<IFORM>(), s, id);
		for (int i = 0; i < n; ++i)
			op(id[i]);
	}

}

/**
 *
 * @param mesh mesh
 * @param points  define region
 *          if points.size() == 1 ,select Nearest Point
 *     else if points.size() == 2 ,select in the rectangle with  diagonal points[0] ~ points[1]
 *     else if points.size() >= 3 && Z<3
 *                    select points in a polyline on the Z-plane whose vertex are points
 *     else if points.size() >= 4 && Z>=3
 *                    select points in a closed surface
 *                    UNIMPLEMENTED
 *     else   illegal input
 *
 * @param fun
 * @param Z  Z==0    polyline on yz-plane
 *           Z==1    polyline on zx-plane,
 *           Z==2    polyline on xy-plane
 *           Z>=3
 */
template<int IFORM, int N, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op,
        std::vector<nTuple<N, Real>> const & points, unsigned int Z = 2)
{

	if (points.size() == 1)
	{

		typename TM::coordinates_type x = { 0, 0, 0 };

		for (int i = 0; i < N; ++i)
		{
			x[(i + Z + 1) % 3] = points[0][i];
		}

		SelectFromMesh<IFORM>(mesh, op, x);
	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		typename TM::coordinates_type v0 = { 0, 0, 0 };
		typename TM::coordinates_type v1 = { 0, 0, 0 };
		for (int i = 0; i < N; ++i)
		{
			v0[(i + Z + 1) % 3] = points[0][i];
			v1[(i + Z + 1) % 3] = points[1][i];
		}
		SelectFromMesh<IFORM>(mesh, op, v0, v1);
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{

		PointInPolygen checkPointsInPolygen(points, Z);

		for (auto s : mesh.GetRange(IFORM))
		{
			auto x = mesh.GetCoordinates(s);

			if (checkPointsInPolygen(x[(Z + 1) % 3], x[(Z + 2) % 3]))
			{
				op(s);
			}
		}

	}
	else
	{
		ERROR << "Illegal input";
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh, std::function<void(typename TM::index_type)> const & op, LuaObject const & dict)
{
	if (dict.is_table())
	{
		std::vector<typename TM::coordinates_type> points;

		dict.as(&points);

		SelectFromMesh<IFORM>(mesh, op, points);

	}
	else if (dict.is_function())
	{

		for (auto s : mesh.GetRange(IFORM))
		{
			auto x = mesh.GetCoordinates(s);

			if (dict(x[0], x[1], x[2]).template as<bool>())
			{
				op(s);
			}
		}
	}

}
}
// namespace simpla

#endif /* SELECT_H_ */
