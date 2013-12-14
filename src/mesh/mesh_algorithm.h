/*
 * mesh_algorithm.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef MESH_ALGORITHM_H_
#define MESH_ALGORITHM_H_
#include "pointinpolygen.h"
namespace simpla
{

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

template<typename TM>
void SelectPointsInRegion(
		std::function<
				void(typename TM::index_type const &,
						typename TM::coordinates_type const &)> const &fun,
		TM const & mesh,
		std::vector<typename TM::coordinates_type> const points,
		unsigned int Z = 2)
{

	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::index_type index_type;

	if (points.size() == 1)
	{
		index_type s = mesh.GetNearestPoints(points[0]);
		fun(s, mesh.GetCoordinates(0, s));
	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		coordinates_type v0 = points[0];
		coordinates_type v1 = points[1];
		TraversalCoordinates(0,

		[&](typename mesh_type::index_type const&s ,
				typename mesh_type:: coordinates_type const &x)
		{
			bool flag=true;
			for(int i=0;i<3;++i)
			{
				if(v0[i]!=v1[i])
				flag&=((v0[i]-x[i])*(x[i]-v1[i])>=0));
			}
			if(flag)
			{
				fun(s,x);
			}
		},

		mesh.WITH_GHOSTS);
	}
	else if (Z < 3) //select points in polyline
	{

		PointInPolygen<typename mesh_type::coordinates_type> checkPointsInPolygen(
				points, Z);

		mesh.TraversalCoordinates(0,

		[&](typename mesh_type::index_type const&s ,
				typename mesh_type:: coordinates_type const &x)
		{
			if(checkPointsInPolygen(x))
			{
				fun(s,x);
			}
		},

		mesh.WITH_GHOSTS);
	}
	else if (points.size() >= 4 && Z >= 3)
	{
		UNIMPLEMENT << " select points in a closed surface";
	}
	else
	{
		ERROR << "Illegal input";
	}

}

}  // namespace simpla

#endif /* MESH_ALGORITHM_H_ */
