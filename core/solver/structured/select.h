/**
 * @file model.h
 *
 *  Created on: 2015-6-24
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_MODEL_H_
#define CORE_MESH_STRUCTURED_MODEL_H_

#include "../../model/model_traits.h"

namespace simpla
{
template<typename ...>
struct Mesh;
namespace tags
{
struct structured;
}  // namespace tags
//
//template<typename CoordinateSystem, typename DistanceFunction, typename TBox,
//		typename ...Args>
//void select(Mesh<CoordinateSystem, tags::structured> const &manifold,
//            DistanceFunction const &dist, TBox const &t_box, Args &&...args)
//{
//	typedef typename Mesh<CoordinateSystem, tags::structured>::id_type id_type;
//	typedef typename Mesh<CoordinateSystem, tags::structured>::topology_type topology_type;
//
//	select(dynamic_cast<topology_type const &>(manifold),
//
//	       [&](id_type t)
//	       {
//	           return static_cast<Real>(dist(manifold.point(t)));
//	       },
//
//	       manifold.inv_map(traits::get<0>(t_box)), manifold.inv_map(traits::get<1>(t_box)),
//
//	       std::forward<Args>(args)...
//
//	);
//
//}

/**
 *
 * @param dist  distance function, which is used to model geometric object
 *   'Real dist(id_type s)' return the nearest orient distance between point(s) and object
 *       > 0 outside , <0 inside , ==0 on the boundary
 *
 * @param op        op(id_type s) , evaluate when point s is selected.
 * @param iform		type of cell, i.e. VERTEX,EDGE,FACE,VOLUME
 * @param s         id of start point
 * @param level     grid level of search region
 * @param tag       define the rule to select
 *     1 inside				, valid for VERTEX,EDGE,FACE,VOLUME
 *     2 outside			, valid for VERTEX,EDGE,FACE,VOLUME
 *     4 boundary - undetermined
 *     5 inside boundary   	, valid for VERTEX,EDGE,FACE
 *     6 outside boundary  	, valid for VERTEX,EDGE,FACE
 *     7 cross boundary    	, valid for         EDGE,FACE,VOLUME
 *     >7 Undetermined
 * @param depth,  if voxel s is not selected then --depth
 * @param SEARCH_DEPTH max depth of searching, which is used to reject
 *      trivial  invalid voxel. When SEARCH_DEPTH is small, it is
 *      possible to miss object smaller then voxel.
 *        suggested value SEARCH_DEPTH=DEFAULT_MESH_RESOLUTION
 */

template<size_t TAGS, typename DistanceFunction, typename OpFunction>
void select(MeshIDs_<TAGS>, DistanceFunction const &dist, OpFunction const &op,
            int select_tag, int iform, typename MeshIDs_<TAGS>::id_type s,
            int level, int node_flag = 0x50000,
            std::shared_ptr<std::map<typename MeshIDs_<TAGS>::id_type, Real>> dist_cache =
            nullptr)
{
	typedef typename MeshIDs_<TAGS>::id_type id_type;

	static constexpr size_t MESH_RESOLUTION = (TAGS & 0xF) % 3;

// FIXME NEED parallel optimize
	if ((node_flag & (~0xFFFF)) == 0) {
		return;
	}

	static constexpr id_type D = 1UL;

	static constexpr id_type DI = D;

	static constexpr id_type DJ = D << (MeshIDs_<TAGS>::ID_DIGITS);

	static constexpr id_type DK = D << (MeshIDs_<TAGS>::ID_DIGITS * 2);

	static constexpr id_type DA = DK | DI | DJ;

	static constexpr id_type m_sibling_node_[8] = {0, DI, DJ, DI | DJ,

	                                               DK, DK | DI, DK | DJ, DK | DJ | DI};

	if (dist_cache == nullptr) {
		dist_cache = std::shared_ptr<std::map<id_type, Real>>(
				new std::map<id_type, Real>);
	}

	for (int i = 0; i < 8; ++i) {
		if ((node_flag & (1UL << (i + 8))) == 0) {
			id_type id = s + (m_sibling_node_[i] << level);

			Real distance = 0;

			auto it = dist_cache->find(id);

			if (it != dist_cache->end()) {
				distance = it->second;
			}
			else {
				distance = dist(id);
				(*dist_cache)[id] = distance;
			}

			if (distance > 0) {
				node_flag |= 1UL << i;
			}
			node_flag |= (1UL << (i + 8));
		}
	}

	bool selected =
			((node_flag & 0xFF) == 0x00 && (select_tag == tags::tag_inside))
			|| ((node_flag & 0xFF) == 0xFF
			    && (select_tag == tags::tag_outside))
			|| ((((node_flag & 0xFF) != 0)
			     && ((node_flag & 0xFF) != 0xFF))
			    && ((select_tag & tags::tag_boundary)
			        == tags::tag_boundary));

	if (level > MESH_RESOLUTION) {
		if (selected) {
			node_flag = (node_flag & 0xFFFF) + 0x50000;
		}
		else {
			node_flag -= 0x10000;
		}
		for (int i = 0; i < 8; ++i) {
			select(dist, op, select_tag, iform,
			       s + (m_sibling_node_[i] << (level - 1)), level - 1,
			       (node_flag & (0xFF0000 | (1UL << i))) | (1UL << (i + 8)),
			       dist_cache);
		}
	}
	else {


		static constexpr id_type sub_cell_num[4] = {8, 12, 6, 1};

		static constexpr id_type sub_cell_flag[4][12] = {

				{0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80},
				{0x03, 0x05, 0x0A, 0x0C, 0x11, 0x22, 0x44, 0x88, 0x30, 0x50, 0xA0, 0xC0},
				{0x0F, 0x33, 0x55, 0xF0, 0xCC, 0xAA},
				{0xFF}

		};

		static constexpr id_type sub_cell_shift[4][12] = {

				{
						0,

						DI << 1,

						DJ << 1,

						(DI | DJ) << 1,

						(DK << 1),

						(DK | DI) << 1,

						(DK | DJ) << 1,

						(DK | DI | DJ) << 1,
				},


				{

						DI,

						DJ,

						DJ | (DI << 1),

						DI | (DJ << 1),

						DK,

						DK | (DI << 1),

						DK | (DJ << 1),

						DK | (DI << 1) | (DJ << 1),

						(DK << 1) | DI,

						(DK << 1) | DJ,

						(DK << 1) | DJ | (DI << 1),

						(DK << 1) | DI | (DJ << 1),

				},
				{

						DI | DJ,

						DI | DK,

						DK | DJ,

						DI | DJ | (DK << 1),

						DI | DK | (DJ << 1),

						DK | DJ | (DI << 1)

				},
				{
						DI | DJ | DK
				}

		};
		if (selected) {
			// get the center of voxel
			for (int i = 0; i < sub_cell_num[iform]; ++i) {
				if (

						(((select_tag & (~tags::tag_boundary)) == tags::tag_outside)
						 && ((node_flag & sub_cell_flag[iform][i])
						     == sub_cell_flag[iform][i]))
						||

						(((select_tag & (~tags::tag_boundary))
						  == tags::tag_inside)
						 && (((~node_flag) & sub_cell_flag[iform][i])
						     == sub_cell_flag[iform][i]))
						||

						(((select_tag & (~tags::tag_boundary))
						  == (tags::tag_inside | tags::tag_outside))
						 && ((node_flag & sub_cell_flag[iform][i]) != 0)
						 && ((node_flag & sub_cell_flag[iform][i])
						     != sub_cell_flag[iform][i]))

						) {
					op(s + (sub_cell_shift[iform][i] << (level - 1)));
				}
			}
		}
	}

}
}  // namespace simpla

#endif /* CORE_MESH_STRUCTURED_MODEL_H_ */
