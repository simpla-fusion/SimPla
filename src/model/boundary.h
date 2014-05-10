/*
 * boundary.h
 *
 *  Created on: 2014年5月6日
 *      Author: salmon
 */

#ifndef BOUNDARY_H_
#define BOUNDARY_H_

namespace simpla
{

template<typename TM>
class Surface
{
public:

	typedef Surface<TM> this_type;

	typedef typename TM mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef nTuple<3, coordinates_type> plane_type;

	template<typename TDict, typename ...Others>
	Surface(TDict const & dict, Others const &...);

	virtual ~Surface();

}
;
}  // namespace simpla

#endif /* BOUNDARY_H_ */
