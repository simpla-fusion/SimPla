/**
 * @file primitive.h
 *
 * @date 2015-6-4
 * @author salmon
 */

#ifndef CORE_GEOMETRY_PRIMITIVE_H_
#define CORE_GEOMETRY_PRIMITIVE_H_

#include "../gtl/type_traits.h"
#include "CoordinateSystem.h"
/**
 *  @ref OpenGIS® Implementation Standard for Geographic information
 *  - Simple feature access -architecture Part 1: Common architecture
 *  @ref boost::geometry GGL
 */
namespace simpla { namespace geometry
{

namespace tags
{
///  dimension 0,1,2,3...
struct simplex;

struct cube;
// dimension 0,1,2,3...
struct box;
// dimension 0,1,2,3...,defined by two point
// 0D
struct point;
// 1D
struct line;
struct spline;
struct spline;

struct triangele;
struct rectangle;
struct circle;
// 3D
struct sphere;
struct torus;
struct cone;
struct cylinder;
struct pyramid;

}  // namespace tags

namespace traits
{

template<typename> struct coordinate_system;
template<typename> struct dimension;
template<typename> struct tag;

template<typename> struct peak;
template<typename> struct ridge;
template<typename> struct facet;

template<typename> struct point_type;
template<typename> struct vector_type;
template<typename> struct length_type;
template<typename CS> struct area_type
{
    typedef typename length_type<CS>::type l_type;
    typedef decltype(std::declval<l_type>() * std::declval<l_type>()) type;
};
template<typename CS> struct volume_type
{
    typedef typename length_type<CS>::type l_type;
    typedef typename area_type<CS>::type a_type;
    typedef decltype(std::declval<a_type>() * std::declval<l_type>()) type;
};

template<typename> struct value_type;
template<typename> struct number_of_points;

template<typename> struct is_chains;
template<typename> struct is_primitive;

}  // namespace traits
namespace model
{
/**
 *  @brief CoordinateChart or geometric primitive , geometric primitive
 *  representing a single, connected, homogeneous element of space
 */
/**
 * @brief Element topological n-dimensional 'geometric primitive' ,
 * Primitive<0> is point (simplex<0>)
 * Primitive<1> is a segment of straight line (simplex<1>) or curve, has to end-point
 * Primitive<2> is triangle (simplex<2>) , rectangle, etc...
 * Primitive<3> is tetrahedron (simplex<3>) , cube , etc...
 */
template<int Dimension, typename ...> struct Primitive;

#define DEF_NTUPLE_OBJECT(_COORD_SYS_, _T_, _NUM_)                                      \
 nTuple<_T_, _NUM_> m_data_;                                                          \
 inline operator nTuple<_T_, _NUM_>()const{ return m_data_; }                              \
 nTuple<_T_, _NUM_> const & as_ntuple()const{return m_data_;}                            \
 nTuple<_T_, _NUM_>   & as_ntuple() {return m_data_;}                                    \
 inline _T_ & operator [](size_t n){ return m_data_[n]; }                             \
 inline _T_ const & operator [](size_t n) const{ return m_data_[n]; }                 \
 template<size_t N> inline _T_ & get(){return m_data_[N]; }                           \
 template<size_t N> inline constexpr _T_ const& get() const  { return m_data_[N]; }

/**
 * @brief Point topological 0-dimensional 'geometric primitive' ,
 *   representing a position
 * @note The boundary of a point is the empty set. [ISO 19107]
 */
template<typename CoordinateSystem>
struct Primitive<0, CoordinateSystem>
{
    typedef typename simpla::geometry::traits::coordinate_type<CoordinateSystem>::type value_type;

    static const size_t ndims = simpla::geometry::traits::dimension<
            CoordinateSystem>::value;

    DEF_NTUPLE_OBJECT(CoordinateSystem, value_type, ndims);
};

template<typename CoordinateSystem>
using Point= Primitive<0, CoordinateSystem>;
template<typename CoordinateSystem>
using LineSegment= Primitive<1, CoordinateSystem>;

/**
 * @brief Vector In geometry, Vector represents the first derivative of 'curve',
 * call element \f$v\in T_P M\f$ 'vectors' at point $P\in M$; $T_P M$ is the 'tagent space'
 * at the point \f$P\f$
 * In code,Vector is the difference type of Point Vector = Point - Point
 */
template<typename CoordinateSystem>
struct Vector
{
    typedef typename simpla::geometry::traits::coordinate_type<CoordinateSystem>::type value_type;

    static const int ndims = simpla::geometry::traits::dimension<
            CoordinateSystem>::value;

    DEF_NTUPLE_OBJECT(CoordinateSystem, value_type, ndims);

};

template<typename OS, typename CoordinateSystem>
OS &operator<<(OS &os, Vector<CoordinateSystem> const &geo)
{
    os << geo.as_ntuple();
    return os;
}

/**
 * @brief CoVector is a linear map from  'vector space'
 *
 */
template<typename CoordinateSystem>
struct CoVector
{
    typedef typename simpla::geometry::traits::coordinate_type<CoordinateSystem>::type value_type;

    static const size_t ndims = simpla::geometry::traits::dimension<
            CoordinateSystem>::value;

    DEF_NTUPLE_OBJECT(CoordinateSystem, value_type, ndims);
};

template<typename OS, typename CoordinateSystem>
OS &operator<<(OS &os, CoVector<CoordinateSystem> const &geo)
{
    os << geo.as_ntuple();
    return os;
}

template<typename CS>
LineSegment<CS> operator-(Point<CS> const &x1, Point<CS> const &x0)
{
    return LineSegment<CS>(x0, x1);
}

template<typename CS>
Point<CS> operator+(Point<CS> const &x0, Vector<CS> const &v)
{
    return Point<CS>(x0.as_ntuple() + v.as_ntuple());
}

template<typename CS>
Vector<CS> operator*(Vector<CS> const &x1, Real a)
{
    Vector<CS> res;
    res.as_ntuple() = (x1.as_ntuple() * a);
    return std::move(res);
}

template<typename CS, typename T2>
Vector<CS> cross(Vector<CS> const &x1, T2 const &v)
{
    Vector<CS> res;
    res.as_ntuple() = cross(x1.as_ntuple(), v);
    return std::move(res);
}

template<typename CS>
auto cross(Vector<CS> const &v0, Vector<CS> const &v1)
DECL_RET_TYPE(inner_product(v0.as_ntuple(), v1.as_ntuple()))

/**
 * THIS is INCOMPLETE!!!
 */
template<typename CoordinateSystem, typename CoordinateType, size_t ...N>
struct Tensor
{
    nTuple<CoordinateType, N...> m_data_;

    inline operator nTuple<CoordinateType, N...>()
    {
        return m_data_;
    }

    inline auto operator[](size_t n)
    DECL_RET_TYPE (m_data_[n])

    inline auto operator[](size_t n) const
    DECL_RET_TYPE (m_data_[n])

    template<size_t M>
    inline auto get()
    DECL_RET_TYPE (m_data_[M])

    template<size_t M>
    inline auto get() const
    DECL_RET_TYPE (m_data_[M])
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct Primitive<Dimension, CoordinateSystem, Tag>
{
    typedef Primitive<Dimension, CoordinateSystem, Tag> this_type;
    typedef typename traits::point_type<this_type>::type vertex_type;
    static constexpr size_t number_of_points = traits::number_of_points<
            this_type>::value;

    DEF_NTUPLE_OBJECT(CoordinateSystem, vertex_type, number_of_points);
};

template<typename OS, size_t Dimension, typename CoordinateSystem, typename Tag>
OS &operator<<(OS &os, Primitive<Dimension, CoordinateSystem, Tag> const &geo)
{
    os << geo.as_ntuple();
    return os;
}

#undef DEF_NTUPLE_OBJECT
}
// namespace model

namespace traits
{

template<size_t Dimension, typename ...Others>
struct is_primitive<model::Primitive<Dimension, Others...>>
{
    static constexpr bool value = true;
};

template<size_t Dimension, typename ...Others>
struct is_chains<model::Primitive<Dimension, Others...>>
{
    static constexpr bool value = false;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct coordinate_system<model::Primitive<Dimension, CoordinateSystem, Tag>>
{
    typedef CoordinateSystem type;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct dimension<model::Primitive<Dimension, CoordinateSystem, Tag>>
{
    static constexpr size_t value = Dimension;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct tag<model::Primitive<Dimension, CoordinateSystem, Tag>>
{
    typedef Tag type;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct point_type<model::Primitive<Dimension, CoordinateSystem, Tag>>
{
    typedef model::Primitive<0, CoordinateSystem, Tag> type;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct value_type<model::Primitive<Dimension, CoordinateSystem, Tag>>
{
    typedef model::Primitive<Dimension, CoordinateSystem, Tag> geo;

    typedef decltype(std::declval<geo>()[0]) type;
};
template<typename CoordinateSystem>
struct number_of_points<model::Primitive<0, CoordinateSystem>>
{
    static constexpr size_t value = 1;
};
template<typename CoordinateSystem>
struct number_of_points<model::Primitive<1, CoordinateSystem>>
{
    static constexpr size_t value = 2;
};

template<size_t Dimension, typename ...Others>
struct peak<model::Primitive<Dimension, Others...>>
{
    typedef typename facet<
            typename ridge<model::Primitive<Dimension, Others...> >::type>::type type;
};

template<size_t Dimension, typename ...Others>
struct ridge<model::Primitive<Dimension, Others...>>
{
    typedef typename facet<
            typename facet<model::Primitive<Dimension, Others...> >::type>::type type;
};

template<size_t Dimension, typename ...Others>
struct facet<model::Primitive<Dimension, Others...>>
{
    typedef model::Primitive<Dimension - 1, Others...> type;
};

} // namespace traits
} // namespace geometry
} // namespace simpla

namespace std
{

template<size_t N, size_t M, typename ... Others>
auto get(simpla::geometry::model::Primitive<M, Others...> &obj)
DECL_RET_TYPE((obj[N]))

template<size_t N, size_t M, typename ...Others>
auto get(simpla::geometry::model::Primitive<M, Others...> const &obj)
DECL_RET_TYPE((obj[N]))

}  // namespace std
#endif /* CORE_GEOMETRY_PRIMITIVE_H_ */
