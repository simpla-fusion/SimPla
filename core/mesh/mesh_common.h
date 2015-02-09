/*
 * @file mesh_common.h
 *
 *  Created on: 2015年1月13日
 *      Author: salmon
 */

#ifndef CORE_MESH_MESH_COMMON_H_
#define CORE_MESH_MESH_COMMON_H_

namespace simpla
{

/**
 * @ingroup diff_geo
 * @{
 */
enum ManifoldTypeID
{
	VERTEX = 0,

	EDGE = 1,

	FACE = 2,

	VOLUME = 3
};
/**
 *  @}
 */
}  // namespace simpla

#endif /* CORE_MESH_MESH_COMMON_H_ */
