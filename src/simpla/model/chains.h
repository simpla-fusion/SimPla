/**
 * @file chains.h
 *
 *  Created on: 2015-6-4
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CHAINS_H_
#define CORE_GEOMETRY_CHAINS_H_

#include <stddef.h>
#include <cstdbool>
#include <map>
#include <vector>

#include "simpla/utilities/mpl.h"
#include "simpla/algebra/nTuple.h"
#include "CoordinateSystem.h"
#include "primitive.h"

namespace simpla {
namespace model {

namespace tags {
struct is_structed;
struct is_closed;
struct is_clockwise;
struct is_unordered;
}  // namespace tags

namespace model {

template<typename ...> struct Chains;

template<typename TPrimitive, typename ... Policies>
struct Chains<TPrimitive, Policies...>
{
    typedef TPrimitive primitive_type;

    typedef typename traits::coordinate_system<primitive_type>::type coordinate_system;

    typedef typename traits::tag<primitive_type>::type tag_type;

    typedef typename traits::point_type<coordinate_system>::type point_type;

    static constexpr size_t max_number_of_points = traits::number_of_points<
            primitive_type>::value;

    static constexpr int dimension = traits::dimension<primitive_type>::value;

    typedef Chains<Primitive<dimension - 1, coordinate_system, tag_type>,
            Policies...> boundary_type;

    typedef size_t id_type;

    typedef nTuple<id_type, max_number_of_points> indices_tuple;

    typedef std::map<id_type, indices_tuple> data_type;

    data_type &data()
    {
        return m_data_;
    }

    data_type const &data() const
    {
        return m_data_;
    }

    boundary_type boundary() const;

private:
    data_type m_data_;

};

template<typename CS, typename ... Policies>
struct Chains<Primitive<1, CS, tags::simplex>, Policies...> : public std::vector<
        Point<CS>>
{
//	typedef Primitive<1, CS, tags::simplex> primitive_type;
//
//	typedef typename traits::coordinate_system<primitive_type>::value_type_info coordinate_system;
//
//	typedef typename traits::GetTag<primitive_type>::value_type_info tag_type;
//
//	typedef typename traits::point_type<coordinate_system>::value_type_info point_type;
//
//	static constexpr size_t max_number_of_points = traits::number_of_points<
//			primitive_type>::entity;
//
//	static constexpr size_t dimension = traits::dimension<primitive_type>::entity;
//
//	typedef Chains<Primitive<dimension - 1, coordinate_system, tag_type>,
//			Policies...> boundary_type;
//
//	typedef size_t mesh_id_type;
//
//	typedef nTuple<mesh_id_type, max_number_of_points> indices_tuple;
//
//	typedef std::map<mesh_id_type, indices_tuple> DataType;
//
//	DataType & m_data()
//	{
//		return m_attr_data_;
//	}
//	DataType const& m_data() const
//	{
//		return m_attr_data_;
//	}
//	boundary_type boundary() const;
//
//private:
//	DataType m_attr_data_;
};

}  // namespace model
namespace traits {

template<typename> struct is_chains;
template<typename> struct is_primitive;
template<typename> struct is_structed;
template<typename> struct closure;
template<typename> struct point_order;

template<typename ...Others>
struct is_primitive<model::Chains<Others...>>
{
    static constexpr bool value = false;
};

template<typename ...Others>
struct is_chains<model::Chains<Others...>>
{
    static constexpr bool value = true;
};

template<typename PrimitiveType, typename ...Others>
struct coordinate_system<model::Chains<PrimitiveType, Others...>>
{
    typedef typename coordinate_system<PrimitiveType>::type type;
};
template<typename PrimitiveType, typename ...Others>
struct dimension<model::Chains<PrimitiveType, Others...>>
{
    static constexpr int value = dimension<PrimitiveType>::value;
};

template<typename PrimitiveType, typename ...Others>
struct is_structed<model::Chains<PrimitiveType, Others...>>
{
    static constexpr bool value = mpl::find_type_in_list<tags::is_structed,
            Others...>::value;
};
template<typename PrimitiveType, typename ...Others>
struct point_order<model::Chains<PrimitiveType, Others...>>
{
    static constexpr int value =
            mpl::find_type_in_list<tags::is_clockwise, Others...>::value ?
            1 :
            (mpl::find_type_in_list<tags::is_unordered, Others...>::value ?
             0 : -1);
};
template<typename PrimitiveType, typename ...Others>
struct closure<model::Chains<PrimitiveType, Others...>>
{
    static constexpr int value =
            mpl::find_type_in_list<tags::is_closed, Others...>::value ? 1 : 0;
};
}  // namespace traits

}  // namespace model
}  // namespace simpla

#endif /* CORE_GEOMETRY_CHAINS_H_ */
