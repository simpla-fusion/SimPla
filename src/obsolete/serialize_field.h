/*
 * serialize_field.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef SERIALIZE_FIELD_H_
#define SERIALIZE_FIELD_H_

namespace simpla
{
template<typename, typename > class Field;
template<typename, int> class Geometry;

template<typename TM, int IFORM, typename TV>
void _Serialize(Field<Geometry<TM, IFORM>, TV> const & f,
		std::string const & url)
{

}

}
// namespace simpla

#endif /* SERIALIZE_FIELD_H_ */
