/*
 * object_wrapper.h
 *
 *  Created on: 2013年11月18日
 *      Author: salmon
 */

#ifndef OBJECT_WRAPPER_H_
#define OBJECT_WRAPPER_H_

#include <engine/object.h>
#include <fetl/field.h>

namespace simpla
{
template<typename T> class ObjectWrapper;
template<typename, typename > class Field;
template<typename, int> class Geometry;

template<typename TM>
struct ObjectWrapper<TM> // Grid
{
	typedef TM mesh_type;
	Object mesh_obj;

	template<int IFORM, typename TV>
	Object CreateField()
	{
		return Object(
				new Field<Geometry<mesh_type, IFORM>, TV>(
						mesh_obj.template as<mesh_type>()

						));
	}
};

}  // namespace simpla

#endif /* OBJECT_WRAPPER_H_ */
