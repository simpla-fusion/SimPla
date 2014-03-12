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
void SelectFromMesh(TM const &mesh,
        std::function<void(typename TM::index_type, typename TM::coordinates_type)> const & op,
        std::function<bool(typename TM::index_type, typename TM::coordinates_type)> const & select)
{
	mesh.template Traversal<IFORM>([&](typename TM:: index_type s )
	{	auto x=mesh.GetCoordinates(s);
		if(select(s,x)) op(s,x);
	});
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
template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh,
        std::function<void(typename TM::index_type, typename TM::coordinates_type)> const & op,
        std::vector<typename TM::coordinates_type> const & points, unsigned int Z = 2)
{

//	if (points.size() == 1)
//	{
////		typename TM::index_type idx = mesh.template GetAdjacentCells(Int2Type<IFORM>(), Int2Type<VERTEX>(),
////		        mesh.GetIndex(points[0]));
////
////		SelectFromMesh<IFORM>(mesh, op, [idx](typename TM::index_type s )->bool
////		{	return (s==idx);});
//
//	}
//	else

	if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		typename TM::coordinates_type v0 = points[0];
		typename TM::coordinates_type v1 = points[1];

		mesh.template Traversal<IFORM>([&](typename TM:: index_type s )
		{	auto x=mesh.GetCoordinates(s);

			if( (((v0[0]-x[0])*(x[0]-v1[0]))>=0)&&
					(((v0[1]-x[1])*(x[1]-v1[1]))>=0)&&
					(((v0[2]-x[2])*(x[2]-v1[2]))>=0)
			)
			{
				op(s,x);
			}
		});
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{

		PointInPolygen checkPointsInPolygen(points, Z);

		mesh.template Traversal<IFORM>([&](typename TM:: index_type s )
		{
			auto x=mesh.GetCoordinates(s);

			if( checkPointsInPolygen(x[(Z+1)%3],x[(Z+2)%3]))
			{
				op(s,x);
			}
		});

	}
	else
	{
		ERROR << "Illegal input";
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh,
        std::function<void(typename TM::index_type, typename TM::coordinates_type)> const & op,
        std::vector<typename TM::index_type> const & idxs)
{
	int M = mesh.template GetNumCompsPerCell<IFORM>();
	for (auto const & s : idxs)
	{
		op(s, mesh.GetCoordinates(IFORM, s));
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh,
        std::function<void(typename TM::index_type, typename TM::coordinates_type)> const & op,
        std::vector<nTuple<TM::NUM_OF_DIMS, size_t>> const & idxs)
{
	int M = mesh.template GetNumCompsPerCell<IFORM>();
	for (auto const & s : idxs)
	{
		for (int m = 0; m < M; ++m)
			op(mesh.GetComponentIndex(IFORM, m, s[0], s[1], s[2]), mesh.GetCoordinates(IFORM, m, s[0], s[1], s[2]));
	}

}

template<int IFORM, typename TM>
void SelectFromMesh(TM const &mesh,
        std::function<void(typename TM::index_type, typename TM::coordinates_type)> const & op, LuaObject const & dict)
{
	if (dict.is_table())
	{
		std::vector<typename TM::coordinates_type> points;
		dict.as(&points);
		SelectFromMesh<IFORM>(mesh, op, points);

	}
	else if (dict.is_function())
	{
		std::vector<typename TM::coordinates_type> points;
		dict.as(&points);
		SelectFromMesh<IFORM>(mesh, op,

		[&]( typename TM::index_type ,typename TM::coordinates_type x)->bool
		{
			return dict(x[0],x[1],x[2]).template as<bool>();
		});
	}

}
}
// namespace simpla

#endif /* SELECT_H_ */
