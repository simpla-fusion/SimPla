/*
 * mesh_algorithm.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef MESH_ALGORITHM_H_
#define MESH_ALGORITHM_H_

#include <vector>

#include "../utilities/log.h"
#include "pointinpolygen.h"

namespace simpla
{
class LuaObject;
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
void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
        std::vector<typename TM::coordinates_type> const & points, unsigned int Z = 2, int flag = TM::WITH_GHOSTS)
{

	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::index_type index_type;

	if (points.size() == 1)
	{
		index_type idx = mesh.GetNearestVertex(points[0]);

		SelectVericsInRegion(mesh, op,

		[idx](index_type s)->bool
		{
			return (s==idx);

		}, flag);

	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		coordinates_type v0 = points[0];
		coordinates_type v1 = points[1];

		SelectVericsInRegion(mesh, op,

		[v0,v1](index_type s, coordinates_type x )->bool
		{
			return (((v0[0]-x[0])*(x[0]-v1[0]))>=0)&&
			(((v0[1]-x[1])*(x[1]-v1[1]))>=0)&&
			(((v0[2]-x[2])*(x[2]-v1[2]))>=0);

		}, flag);
	}
	else if (Z < 3) //select points in polyline
	{

		PointInPolygen<typename mesh_type::coordinates_type> checkPointsInPolygen(points, Z);

		SelectVericsInRegion(mesh, op, [&](index_type s, coordinates_type x )->bool
		{	return checkPointsInPolygen(x);}, flag);

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

template<typename TM>
void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
        std::function<bool(typename TM::index_type, typename TM::coordinates_type const &)> const & select,
        int flag = 0)
{
	typedef TM mesh_type;
	mesh.Traversal(0,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(s,x), s);
	},

	flag);

}

template<typename TM>
void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
        std::function<bool(typename TM::index_type)> const & select, int flag = 0)
{
	typedef TM mesh_type;
	mesh.Traversal(0,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(s), s);
	},

	flag);

}

template<typename TM>
void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
        std::function<bool(typename TM::coordinates_type const &)> const & select, int flag = 0)
{
	typedef TM mesh_type;
	mesh.Traversal(0,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(x), s);
	},

	flag);

}

template<typename TM>
void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
        LuaObject const & select, int flag = 0)
{
	typedef TM mesh_type;
	mesh.Traversal(0,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(x[0],x[1],x[2]).template as<bool>(), s);
	},

	flag);

}

//template<typename TM, typename ...Args>
//void SelectVericsInRegion(TM const & mesh, std::function<void(bool, typename TM::index_type const &)> const & op,
//        std::function<bool(Args const &...)> const & select, int flag = 0)
//{
//	typedef TM mesh_type;
//	mesh.Traversal(0,
//
//	[&](typename mesh_type::index_type const&s ,
//			typename mesh_type::coordinates_type const &x)
//	{
//		op(select(std::forward<Args const&>(args)...), s);
//	},
//
//	flag);
//
//}

//namespace _impl
//{
//enum
//{
//	PARALLEL, PERPENDICULAR
//};
//
//typedef typename mesh_type::index_type index_type;
//template<int DIRECTION>
//void _SetInterface(Int2Type<0>, Int2Type<DIRECTION>, index_type s0, tag_type in, tag_type const* v,
//        std::set<index_type> *res)
//{
//	if (v[0] == in)
//		res->insert((s0));
//}
//void _SetInterface(Int2Type<1>, Int2Type<PARALLEL>, index_type s0, tag_type in, tag_type const* v,
//        std::set<index_type> *res)
//{
//	if ((v[0] == in) && (v[0] == v[1]))
//		res->insert((s0) * 3 + 0);
//
//	if ((v[2] == v[3]) && (v[3] == in))
//		res->insert((s0 + strides_[1]) * 3 + 0);
//
//	if ((v[4] == v[5]) && (v[4] == in))
//		res->insert((s0 + strides_[2]) * 3 + 0);
//
//	if ((v[6] == v[7]) && (v[7] == in))
//		res->insert((s0 + strides_[2] + strides_[1]) * 3 + 0);
//
//	//
//
//	if ((v[0] == v[2]) && (v[2] == in))
//		res->insert((s0) * 3 + 1);
//
//	if ((v[1] == v[3]) && (v[1] == in))
//		res->insert((s0 + strides_[0]) * 3 + 1);
//
//	if ((v[4] == v[6]) && (v[6] == in))
//		res->insert((s0 + strides_[2]) * 3 + 1);
//
//	if ((v[5] == v[7]) && (v[5] == in))
//		res->insert((s0 + strides_[2] + strides_[0]) * 3 + 1);
//
//	//
//
//	if ((v[0] == v[4]) && (v[0] == in))
//		res->insert((s0) * 3 + 2);
//
//	if ((v[1] == v[5]) && (v[1] == in))
//		res->insert((s0 + strides_[0]) * 3 + 2);
//
//	if ((v[2] == v[6]) && (v[2] == in))
//		res->insert((s0 + strides_[1]) * 3 + 2);
//
//	if ((v[3] == v[7]) && (v[3] == in))
//		res->insert((s0 + strides_[0] + strides_[1]) * 3 + 2);
//
//}
//void _SetInterface(Int2Type<1>, Int2Type<PERPENDICULAR>, index_type s0, tag_type in, tag_type const* v,
//        std::set<index_type> *res)
//{
//	if ((v[0] != v[1]))
//		res->insert((s0) * 3 + 0);
//
//	if ((v[2] != v[3]))
//		res->insert((s0 + strides_[1]) * 3 + 0);
//
//	if ((v[4] != v[5]))
//		res->insert((s0 + strides_[2]) * 3 + 0);
//
//	if ((v[6] != v[7]))
//		res->insert((s0 + strides_[2] + strides_[1]) * 3 + 0);
//
//	//
//
//	if ((v[0] != v[2]))
//		res->insert((s0) * 3 + 1);
//
//	if ((v[1] != v[3]))
//		res->insert((s0 + strides_[0]) * 3 + 1);
//
//	if ((v[4] != v[6]))
//		res->insert((s0 + strides_[2]) * 3 + 1);
//
//	if ((v[5] != v[7]))
//		res->insert((s0 + strides_[2] + strides_[0]) * 3 + 1);
//
//	//
//
//	if ((v[0] != v[4]))
//		res->insert((s0) * 3 + 2);
//
//	if ((v[1] != v[5]))
//		res->insert((s0 + strides_[0]) * 3 + 2);
//
//	if ((v[2] != v[6]))
//		res->insert((s0 + strides_[1]) * 3 + 2);
//
//	if ((v[3] != v[7]))
//		res->insert((s0 + strides_[0] + strides_[1]) * 3 + 2);
//
//}
//
//void _SetInterface(Int2Type<2>, Int2Type<PARALLEL>, index_type s0, tag_type in, tag_type const* v,
//        std::set<index_type> *res)
//{
//
//	if (!((v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3])))
//		res->insert((s0) * 3 + 2);
//	if (!((v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7])))
//		res->insert((s0 + strides_[2]) * 3 + 2);
//
//	if (!((v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5])))
//		res->insert((s0) * 3 + 1);
//	if (!((v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7])))
//		res->insert((s0 + strides_[1]) * 3 + 1);
//
//	if (!((v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6])))
//		res->insert((s0) * 3 + 0);
//	if (!((v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7])))
//		res->insert((s0 + strides_[0]) * 3 + 1);
//
//}
//
//void _SetInterface(Int2Type<2>, Int2Type<PERPENDICULAR>, index_type s0, tag_type in, tag_type const* v,
//        std::set<index_type> *res)
//{
//
//	if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3]))
//		res->insert((s0) * 3 + 2);
//	if ((v[4] == in) && (v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7]))
//		res->insert((s0 + strides_[2]) * 3 + 2);
//
//	if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5]))
//		res->insert((s0) * 3 + 1);
//	if ((v[2] == in) && (v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7]))
//		res->insert((s0 + strides_[1]) * 3 + 1);
//
//	if ((v[0] == in) && (v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6]))
//		res->insert((s0) * 3 + 0);
//	if ((v[1] == in) && (v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7]))
//		res->insert((s0 + strides_[0]) * 3 + 1);
//
//}
//
//void _SetInterface(Int2Type<3>, index_type s0, tag_type in, tag_type const* v, std::set<index_type> *res)
//{
//
//	if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[2]) && (v[2] == v[3]))
//		res->insert((s0 - strides_[2]));
//	if ((v[4] == in) && (v[4] == v[5]) && (v[5] == v[6]) && (v[6] == v[7]))
//		res->insert((s0 + strides_[2]));
//
//	if ((v[0] == in) && (v[0] == v[1]) && (v[1] == v[4]) && (v[4] == v[5]))
//		res->insert((s0 - strides_[1]));
//	if ((v[2] == in) && (v[2] == v[3]) && (v[3] == v[6]) && (v[6] == v[7]))
//		res->insert((s0 + strides_[1]));
//
//	if ((v[0] == in) && (v[0] == v[2]) && (v[2] == v[4]) && (v[4] == v[6]))
//		res->insert((s0 + strides_[0]));
//	if ((v[1] == in) && (v[1] == v[3]) && (v[3] == v[5]) && (v[5] == v[7]))
//		res->insert((s0 + strides_[0]));
//
//	WARNING << "This implement is incorrect when the boundary has too sharp corner";
//
//	/**
//	 *  FIXME this is incorrect when the boundary has too sharp corner
//	 *  for example
//	 *                  ^    out
//	 *                 / \
//	 *       @--------/-@-\----------@
//	 *       |       /  |  \         |
//	 *       |      /   |   \        |
//	 *       |     /    |    \       |
//	 *       |    /     |     \      |
//	 *       @---/------@------\-----@
//	 *          /               \
//		 *         /       in        \
//		 */
//
//}
//
//}  // namespace _impl
//
//template<typename TM, int IFORM, int DIRECTION>
//void SelectPointsOnInterface(TM const &mesh, int IForm,
//
//typename TM::Container<typename TM::tag_type> const & tags,
//
//typename TM::tag_type in, typename TM::tag_type out,
//
//std::vector<typename TM::index_type>*res) const
//{
//	typedef TM mesh_type;
//	typedef typename mesh_type::index_type index_type;
//	typedef typename mesh_type::tag_type tag_type;
//
//	auto const & dims_ = mesh.GetDemensions();
//	auto const & strides_ = mesh.GetStrides();
//
//	constexpr int num_verte_per_cell = mesh_type::MAX_NUM_VERTEX_PER_CEL;
//
//	std::set<index_type> tmp_res;
//
//	std::vector<tag_type> mtags;
//
//	mesh.TraversalIndex(IForm, [&](int m,index_type const & s)
//	{
//		tag_type v[num_verte_per_cell];
//
//		/** @ref The Visualization Toolkit 4th Ed. p.273
//		 *
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *          / |             / |
//		 *         /  |            /  |
//		 *        4---|-----------5   |
//		 *        | --> B0        |   |
//		 *        |   2-----------|---3
//		 *        E2 /    ^B2       |  /
//		 *        | E1    |     | /
//		 *        |/              |/
//		 *        0------E0-------1   ---> x
//		 *
//		 *
//		 */
//
//		mesh.GetComponents(v,m,s);
//
//		// not interface
//		    if (((v[0] = v[1]) && (v[1] = v[2]) && (v[2] = v[3]) && (v[3] = v[4]) && (v[4] = v[5]) && (v[5] = v[6])
//						    && (v[6] = v[7]))
//
//				    || ((v[0] != in) && (v[1] != in) && (v[2] != in) && (v[3] != in) && (v[4] != in) && (v[5] != in)
//						    && (v[6] != in) && (v[7] != in))
//
//				    || ((v[0] != out) && (v[1] != out) && (v[2] != out) && (v[3] != out) && (v[4] != out)
//						    && (v[5] != out) && (v[6] != out) && (v[7] != out)))
//		    continue;
//
////		    _SetInterface(Int2Type<IFORM>(), Int2Type<DIRECTION>(), s, in, v, &tmp_res);
//
//	    });
//	res->clear();
//	std::copy(tmp_res.begin(), tmp_res.end(), std::back_inserter(*res));
//
//}

}
// namespace simpla

#endif /* MESH_ALGORITHM_H_ */
