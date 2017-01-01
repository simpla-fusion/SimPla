/**
 * @file  calculate_fvm.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef CALCULATE_FVM_H_
#define CALCULATE_FVM_H_

#include <complex>
#include <cstddef>
#include <type_traits>

#include <simpla/toolbox/sp_def.h>
#include <simpla/mpl/macro.h>
#include <simpla/mpl/type_traits.h>
#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/Expression.h>
#include <simpla/algebra/Calculus.h>


namespace simpla { namespace algebra
{

namespace declare { template<typename, typename, size_type ...I> struct Field_; }
namespace st= simpla::traits;
namespace calculus
{
using namespace mesh;

/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
template<typename TM>
struct InterpolatePolicy
{
    typedef TM mesh_type;
    typedef InterpolatePolicy<mesh_type> this_type;
    typedef mesh::MeshEntityIdCoder M;
public:
    InterpolatePolicy() {}

    virtual ~InterpolatePolicy() {}


private:
    template<typename U, typename M, size_type...I> inline U const &
    eval(declare::Field_<U, M, I...> &f, MeshEntityId const &s) { return f[s]; };

    template<typename U, typename M, size_type...I> inline U &
    eval(declare::Field_<U, M, I...> const &f, MeshEntityId const &s) { return f[s]; };

    template<typename TD, typename TIDX>
    static inline traits::value_type_t <TD>
    gather_impl_(TD const &f, TIDX const &idx)
    {

        MeshEntityId X = (M::_DI);
        MeshEntityId Y = (M::_DJ);
        MeshEntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        MeshEntityId s = std::get<0>(idx);

        return eval(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +//
               eval(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +//
               eval(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +//
               eval(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +//
               eval(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +//
               eval(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +//
               eval(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +//
               eval(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

public:

    template<typename TF>
    constexpr static inline traits::field_value_t <TF>
    gather(mesh_type const &m, TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == VERTEX)))
    {
        return gather_impl_(f, m.point_global_to_local(r, 0));
    }

    template<typename TF>
    constexpr static inline traits::field_value_t <TF>
    gather(mesh_type const &m, TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == EDGE)))
    {
        return traits::field_value_t<TF> {
                gather_impl_(f, m.point_global_to_local(r, 1)),
                gather_impl_(f, m.point_global_to_local(r, 2)),
                gather_impl_(f, m.point_global_to_local(r, 4))
        };
    }

    template<typename TF>
    constexpr static inline traits::field_value_t <TF>
    gather(mesh_type const &m, TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == FACE)))
    {
        return traits::field_value_t<TF> {
                gather_impl_(f, m.point_global_to_local(r, 6)),
                gather_impl_(f, m.point_global_to_local(r, 5)),
                gather_impl_(f, m.point_global_to_local(r, 3))
        };
    }

    template<typename TF>
    constexpr static inline traits::field_value_t <TF>
    gather(mesh_type const &m, TF const &f, point_type const &x, ENABLE_IF((traits::iform<TF>::value == VOLUME)))
    {
        return gather_impl_(f, m.point_global_to_local(x, 7));
    }


private:
    template<typename TF, typename IDX, typename TV>
    static inline void
    scatter_impl_(TF &f, IDX const &idx,
                  TV const &v)
    {

        MeshEntityId X = (M::_DI);
        MeshEntityId Y = (M::_DJ);
        MeshEntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        MeshEntityId s = std::get<0>(idx);

        eval(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        eval(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        eval(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        eval(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        eval(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        eval(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        eval(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        eval(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

    }


    template<typename TF, typename TX, typename TV>
    static inline void
    scatter_(mesh_type const &m, index_const <VERTEX>, TF &
    f, TX const &x, TV const &u)
    {
        scatter_impl_(f, m.point_global_to_local(x, 0), u);
    }

    template<typename TF, typename TX, typename TV>
    static inline void
    scatter_(mesh_type const &m, index_const <EDGE>, TF &
    f, TX const &x, TV const &u)
    {

        scatter_impl_(f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 4), u[2]);

    }

    template<typename TF, typename TX, typename TV>
    static inline void
    scatter_(mesh_type const &m, index_const <FACE>, TF &f,
             TX const &x, TV const &u)
    {

        scatter_impl_(f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 3), u[2]);
    }

    template<typename TF, typename TX, typename TV>
    static inline void
    scatter_(mesh_type const &m, index_const <VOLUME>,
             TF &f, TX const &x, TV const &u)
    {
        scatter_impl_(f, m.point_global_to_local(x, 7), u);
    }

public:
    template<typename TF, typename ...Args>
    static inline void
    scatter(mesh_type const &m, TF &f, Args &&...args)
    {
        scatter_(m, traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

private:
    template<typename TV>
    static inline TV
    sample_(mesh_type const &m, index_const <VERTEX>, MeshEntityId const &s,
            TV const &v) { return v; }

    template<typename TV, size_type L> static inline TV
    sample_(mesh_type const &m, index_const <VERTEX>, MeshEntityId const &s,
            nTuple <TV, L> const &v) { return v[s.w % L]; }

    template<typename TV>
    static inline TV
    sample_(mesh_type const &m, index_const <VOLUME>, MeshEntityId const &s,
            TV const &v) { return v; }

    template<typename TV, size_type L> static inline TV
    sample_(mesh_type const &m, index_const <VOLUME>, MeshEntityId const &s,
            nTuple <TV, L> const &v) { return v[s.w % L]; }

    template<typename TV>
    static inline TV
    sample_(mesh_type const &m, index_const <EDGE>, MeshEntityId const &s, nTuple<TV, 3> const &v)
    {
        return v[M::sub_index(s)];
    }

    template<typename TV>
    static inline TV
    sample_(mesh_type const &m, index_const <FACE>, MeshEntityId const &s, nTuple<TV, 3> const &v)
    {
        return v[M::sub_index(s)];
    }
//
//    template<typename M,int IFORM,  typename TV>
//    static inline  TV sample_(M const & m,index_const< IFORM>, mesh_id_type s,
//                                       TV const &v) { return v; }

public:

//    template<typename M,int IFORM,  typename TV>
//    static inline  auto generate(TI const &s, TV const &v)
//    DECL_RET_TYPE((sample_(M const & m,index_const< IFORM>(), s, v)))


    template<int IFORM, typename TV>
    static inline traits::value_type_t <TV>
    sample(mesh_type const &m, MeshEntityId const &s, TV const &v)
    {
        return sample_(m, index_const<IFORM>(), s, v);
    }
//    DECL_RET_TYPE((sample_(index_const< IFORM>(), s, v)))


    /**
     * A radial basis function (RBF) is a real-valued function whose value depends only
     * on the distance from the origin, so that \f$\phi(\mathbf{x}) = \phi(\|\mathbf{x}\|)\f$;
     * or alternatively on the distance from some other point c, called a center, so that
     * \f$\phi(\mathbf{x}, \mathbf{c}) = \phi(\|\mathbf{x}-\mathbf{c}\|)\f$.
     */

    Real RBF(mesh_type const &m, point_type const &x0, point_type const &x1, vector_type const &a)
    {
        vector_type r;
        r = (x1 - x0) / a;
        // @NOTE this is not  an exact  RBF
        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
    }


    Real RBF(mesh_type const &m, point_type const &x0, point_type const &x1, Real const &a)
    {
        return (1.0 - m.distance(x1, x0) / a);
    }

    template<typename V, mesh::MeshEntityType IFORM, size_type DOF, typename U> static inline void
    assign(declare::Field_<V, mesh_type, IFORM, DOF> &f, mesh_type const &m, MeshEntityId const &s,
           nTuple <U, DOF> const &v)
    {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
    }

    template<typename V, size_type DOF, typename U> static inline void
    assign(declare::Field_<V, mesh_type, EDGE, DOF> &f, mesh_type const &m, MeshEntityId const &s,
           nTuple<U, 3> const &v)
    {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
    }

    template<typename V, size_type DOF, typename U> static inline void
    assign(declare::Field_<V, mesh_type, FACE, DOF> &f, mesh_type const &m, MeshEntityId const &s,
           nTuple<U, 3> const &v)
    {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
    }

    template<typename V, size_type DOF, typename U> static inline void
    assign(declare::Field_<V, mesh_type, VOLUME, DOF> &f, mesh_type const &m, MeshEntityId const &s,
           nTuple <U, DOF> const &v)
    {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
    }

    template<typename V, mesh::MeshEntityType IFORM, size_type DOF, typename U> static inline void
    assign(declare::Field_<V, mesh_type, IFORM, DOF> &f, mesh_type const &m, MeshEntityId const &s, U const &v)
    {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v; }
    }


};


/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct calculator<declare::Field_<TV, TM, IFORM, DOF> >
{
    typedef calculator<declare::Field_<TV, TM, IFORM, DOF> > this_type;
    typedef declare::Field_<TV, TM, IFORM, DOF> self_type;
    typedef TV value_type;
    typedef TM mesh_type;
    typedef mesh::MeshEntityIdCoder M;
    typedef mesh::MeshEntityId MeshEntityId;


    typedef declare::Array_<TV,
            traits::rank<TM>::value +
            ((IFORM == VERTEX || IFORM == VOLUME ? 0 : 1) * DOF > 1 ? 1 : 0)> data_block_type;

    static std::shared_ptr<data_block_type> create_data_block(TM const *m)
    {
        auto dims = m->mesh_block()->dimensions();
        size_type d[4] = {dims[0], dims[1], dims[2], 0};
        return std::make_shared<data_block_type>(d);
    }

    static void
    deploy(self_type &self)
    {
        if (self.m_data_holder_.get() == nullptr) { self.m_data_holder_ = create_data_block(self.m_mesh_); }

        self.m_data_ = self.m_data_holder_.get();

        self.m_data_->deploy();

    };

    static void
    reset(self_type &self)
    {
        self.m_data_ = nullptr;
        self.m_data_holder_.reset();
    };

    static void
    clear(self_type &self)
    {
        deploy(self);
        self.m_data_->clear();
    };

/********************************************************************************************************/
    inline static value_type const &
    get_value(mesh_type const &m, data_block_type const &d, MeshEntityId const &s)
    {
        auto idx = M::unpack_index4(s);
        return d.at(&idx[0]);
    };

    inline static value_type &
    get_value(mesh_type const &m, data_block_type &d, MeshEntityId const &s)
    {
        auto idx = M::unpack_index4(s);
        return d.at(&idx[0]);
    };


    template<typename U, typename M, size_type...I> inline static U const &
    get_value(mesh_type const &m, declare::Field_<U, M, I...> const &f,
              MeshEntityId const &s)
    {
        return get_value(m, *f.data(), s);
    };

    template<typename U, typename M, size_type...I> inline static U &
    get_value(mesh_type const &m, declare::Field_<U, M, I...> &f, MeshEntityId const &s)
    {
        return get_value(m, *f.data(), s);
    };


private:

    template<typename FExpr> inline static traits::value_type_t <FExpr>
    get_v(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return get_value(m, f, s) * m.volume(s); }

    template<typename FExpr> inline static traits::value_type_t <FExpr>
    get_d(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return get_value(m, f, s) * m.dual_volume(s); }

public:

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_exterior_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_exterior_derivative, T> const &f,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == VERTEX))
    {
        MeshEntityId D = M::delta_index(s);
        return (get_v(m, std::get<0>(f.m_args_), s + D) - get_v(m, std::get<0>(f.m_args_), s - D)) * m.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_exterior_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_exterior_derivative, T> const &expr,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == EDGE))
    {

        MeshEntityId X = M::delta_index(M::dual(s));
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);


        return ((get_v(m, std::get<0>(expr.m_args_), s + Y) - get_v(m, std::get<0>(expr.m_args_), s - Y))
                - (get_v(m, std::get<0>(expr.m_args_), s + Z) - get_v(m, std::get<0>(expr.m_args_), s - Z))
               ) * m.inv_volume(s);


    }

    //! div<1>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_codifferential_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_codifferential_derivative, T> const &expr,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == EDGE))
    {


        return -(get_d(m, std::get<0>(expr.m_args_), s + M::_DI) - get_d(m, std::get<0>(expr.m_args_), s - M::_DI)
                 + get_d(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_d(m, std::get<0>(expr.m_args_), s - M::_DJ)
                 + get_d(m, std::get<0>(expr.m_args_), s + M::_DK) - get_d(m, std::get<0>(expr.m_args_), s - M::_DK)
        ) * m.inv_dual_volume(s);
    }

    //! div<2>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_exterior_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_exterior_derivative, T> const &expr,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == FACE))
    {
        return (get_v(m, std::get<0>(expr.m_args_), s + M::_DI) - get_v(m, std::get<0>(expr.m_args_), s - M::_DI)
                + get_v(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_v(m, std::get<0>(expr.m_args_), s - M::_DJ)
                + get_v(m, std::get<0>(expr.m_args_), s + M::_DK) - get_v(m, std::get<0>(expr.m_args_), s - M::_DK)
               ) * m.inv_volume(s);

    }


    //! curl<2>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_codifferential_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_codifferential_derivative, T> const &expr,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == FACE))
    {

        MeshEntityId X = M::delta_index(s);
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);

        return -((get_d(m, std::get<0>(expr.m_args_), s + Y) - get_d(m, std::get<0>(expr.m_args_), s - Y))
                 - (get_d(m, std::get<0>(expr.m_args_), s + Z) - get_d(m, std::get<0>(expr.m_args_), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_codifferential_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_codifferential_derivative, T> const &expr,
              MeshEntityId const &s, ENABLE_IF(traits::iform<T>::value == VOLUME))
    {
        MeshEntityId D = M::delta_index(M::dual(s));
        return -(get_d(m, std::get<0>(expr.m_args_), s + D) - get_d(m, std::get<0>(expr.m_args_), s - D)) *
               m.inv_dual_volume(s);
    }

    template<typename T>
    static inline traits::value_type_t <declare::Expression<tags::_codifferential_derivative, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_codifferential_derivative, T> const &expr,
              MeshEntityId const &s)
    {
        static_assert(traits::iform<T>::value != VOLUME && traits::iform<T>::value != VERTEX,
                      "ERROR: grad VERTEX/VOLUME Field  ");
    };
    //! *Form<IR> => Form<N-IL>


    template<typename T, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::_hodge_star, T>>
    get_value(mesh_type const &m, declare::Expression<tags::_hodge_star, T> const &expr,
              MeshEntityId const &s)
    {
        auto const &l = std::get<0>(expr.m_args_);
        int i = M::iform(s);
        MeshEntityId X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(M::dual(s));
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);


        return (
                       get_v(m, l, ((s - X) - Y) - Z) +
                       get_v(m, l, ((s - X) - Y) + Z) +
                       get_v(m, l, ((s - X) + Y) - Z) +
                       get_v(m, l, ((s - X) + Y) + Z) +
                       get_v(m, l, ((s + X) - Y) - Z) +
                       get_v(m, l, ((s + X) - Y) + Z) +
                       get_v(m, l, ((s + X) + Y) - Z) +
                       get_v(m, l, ((s + X) + Y) + Z)

               ) * m.inv_dual_volume(s) * 0.125;


    };

    ////***************************************************************************************************
    //! p_curl<1>
    static constexpr Real m_p_curl_factor_[3] = {0, 1, -1};

    template<typename T, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::_p_exterior_derivative < I>, T>>
    get_value(
    mesh_type const &m, declare::Expression<tags::_p_exterior_derivative < I>, T
    > const &expr,
    MeshEntityId const &s,
    ENABLE_IF(traits::iform<T>::value == EDGE)
    )
    {
        return (get_v(m, std::get<0>(expr.m_args_), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.m_args_), s - M::DI(I))
               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }


    template<typename T, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::_p_codifferential_derivative < I>, T>>
    get_value(
    mesh_type const &m, declare::Expression<tags::_p_codifferential_derivative < I>, T
    > const &expr,
    MeshEntityId const &s,
    ENABLE_IF(traits::iform<T>::value == FACE)
    )
    {

        return (get_v(m, std::get<0>(expr.m_args_), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.m_args_), s - M::DI(I))
               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }

////***************************************************************************************************
//
////! map_to
//    template<typename T, size_t I>
//    inline static T
//    _map_to(mesh_type const &m, T const &r, MeshEntityId const &s, index_sequence<VERTEX, I>,
//          st::is_primary_t<T> *_p = nullptr) { return r; }
//
//    template<typename TF, size_t I>
//    inline static traits::value_type_t<TF>
//    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<I, I>,
//          std::enable_if_t<!st::is_primary<TF>::value>
//          *_p = nullptr) { return get_value(m, expr, s); }
private:
    template<size_type I, typename T> inline static typename traits::value_type<T>::type
    _map_to(mesh_type const &m, T const &f, MeshEntityId const &s, index_sequence <I, I>) { return f[s]; };

    template<typename TF> static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VERTEX, EDGE>)
    {
        int n = M::sub_index(s);
        MeshEntityId X = M::delta_index(s);
        auto l = get_value(m, expr, sw(s - X, n));
        auto r = get_value(m, expr, sw(s + X, n));
        return (l + r) * 0.5;
//        return (get_value(m, expr, sw(s - X, n)) + get_value(m, expr, sw(s + X, n))) * 0.5;


    }


    template<typename TF> static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VERTEX, FACE>)
    {
        int n = M::sub_index(s);

        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       get_value(m, expr, sw(s - Y - Z, n)) +
                       get_value(m, expr, sw(s - Y + Z, n)) +
                       get_value(m, expr, sw(s + Y - Z, n)) +
                       get_value(m, expr, sw(s + Y + Z, n))
               ) * 0.25;
    }


    template<typename TF> static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VERTEX, VOLUME>)
    {
        auto const &l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (get_value(m, l, s - X - Y - Z) +
                get_value(m, l, s - X - Y + Z) +
                get_value(m, l, s - X + Y - Z) +
                get_value(m, l, s - X + Y + Z) +
                get_value(m, l, s + X - Y - Z) +
                get_value(m, l, s + X - Y + Z) +
                get_value(m, l, s + X + Y - Z) +
                get_value(m, l, s + X + Y + Z)
               ) * 0.125;
    }


    template<typename TF> static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <EDGE, VERTEX>)
    {
        MeshEntityId X = M::DI(s.w, s);
        return (get_value(m, expr, sw(s - X, 0)) + get_value(m, expr, sw(s + X, 0))) * 0.5;
    }


    template<typename TF> static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <FACE, VERTEX>)
    {


        MeshEntityId Y = M::DI((s.w + 1) % 3, s);
        MeshEntityId Z = M::DI((s.w + 2) % 3, s);

        return (get_value(m, expr, sw(s - Y - Z, 0)) +
                get_value(m, expr, sw(s - Y + Z, 0)) +
                get_value(m, expr, sw(s + Y - Z, 0)) +
                get_value(m, expr, sw(s + Y + Z, 0))) * 0.25;

//        MeshEntityId X = M::_DI;
//        MeshEntityId Y = M::_DJ;
//        MeshEntityId Z = M::_DK;
//        return nTuple<typename traits::value_type<TF>::type, 3>
//                {
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - Y - Z)) +
//                                                                           get_value(m, l, (s - Y + Z)) +
//                                                                           get_value(m, l, (s + Y - Z)) +
//                                                                           get_value(m, l, (s + Y + Z)) * 0.25),
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - Z - X)) +
//                                                                           get_value(m, l, (s - Z + X)) +
//                                                                           get_value(m, l, (s + Z - X)) +
//                                                                           get_value(m, l, (s + Z + X)) * 0.25),
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - X - Y)) +
//                                                                           get_value(m, l, (s - X + Y)) +
//                                                                           get_value(m, l, (s + X - Y)) +
//                                                                           get_value(m, l, (s + X + Y)) * 0.25)
//                };


    }


    template<typename TF>
    static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VOLUME, VERTEX>)
    {
        auto const &l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (
                       get_value(m, l, ((s - X - Y - Z))) +
                       get_value(m, l, ((s - X - Y + Z))) +
                       get_value(m, l, ((s - X + Y - Z))) +
                       get_value(m, l, ((s - X + Y + Z))) +
                       get_value(m, l, ((s + X - Y - Z))) +
                       get_value(m, l, ((s + X - Y + Z))) +
                       get_value(m, l, ((s + X + Y - Z))) +
                       get_value(m, l, ((s + X + Y + Z)))
               ) * 0.125;
    }


    template<typename TF>
    static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VOLUME, FACE>)
    {
        auto X = M::delta_index(M::dual(s));

        return (get_value(m, expr, s - X) + get_value(m, expr, s + X)) * 0.5;
    }


    template<typename TF>
    static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <VOLUME, EDGE>)
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       get_value(m, l, s - Y - Z) +
                       get_value(m, l, s - Y + Z) +
                       get_value(m, l, s + Y - Z) +
                       get_value(m, l, s + Y + Z)
               ) * 0.25;
    }


    template<typename TF>
    static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <FACE, VOLUME>)
    {

        MeshEntityId X = M::DI(s.w, s);

        return (get_value(m, expr, sw(s - X, 0)) + get_value(m, expr, sw(s + X, 0))) * 0.5;


    }


    template<typename TF>
    static inline traits::value_type_t <TF>
    _map_to(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence <EDGE, VOLUME>)
    {
//        auto const &l = expr;
//
//        auto X = M::DI(0, s);
//        auto Y = M::DI(1, s);
//        auto Z = M::DI(2, s);

        MeshEntityId Y = M::DI((s.w + 1) % 3, s);
        MeshEntityId Z = M::DI((s.w + 1) % 3, s);

        return (get_value(m, expr, sw(s - Y - Z, 0)) + get_value(m, expr, sw(s - Y + Z, 0)) +
                get_value(m, expr, sw(s + Y - Z, 0)) + get_value(m, expr, sw(s + Y + Z, 0))) * 0.25;


    }

public:
    template<size_type I, typename T>
    static inline traits::value_type_t <T>
    map_to(mesh_type const &m, T const &expr, MeshEntityId const &s)
    {
        return _map_to(m, expr, s, index_sequence<traits::iform<T>::value, I>());
    };

    template<size_type I, typename T>
    static inline traits::value_type_t <T>
    get_value(mesh_type const &m, declare::Expression<tags::_map_to < I>, T

    > const &expr,
    MeshEntityId const &s
    )
    {
        return map_to<I>(m, std::get<0>(expr.m_args_), s);
    };

    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, size_t IL, size_t IR>
    static inline traits::value_type_t <declare::Expression<tags::_wedge, T...>>
    get_value(mesh_type const &m, declare::Expression<tags::_wedge, T...> const &expr,
              MeshEntityId const &s, index_sequence <IL, IR>)
    {
        return m.inner_product(_map_to(m, std::get<0>(expr.m_args_), s, index_sequence<IL, IR + IL>()),
                               _map_to(m, std::get<1>(expr.m_args_), s, index_sequence<IR, IR + IL>()),
                               s);

    }


    template<typename TL, typename TR>
    static inline traits::value_type_t <declare::Expression<tags::_wedge, TL, TR>>
    get_value(mesh_type const &m, declare::Expression<tags::_wedge, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence <EDGE, EDGE>)
    {
        auto const &l = std::get<0>(expr.m_args_);
        auto const &r = std::get<1>(expr.m_args_);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((get_value(m, l, s - Y) + get_value(m, l, s + Y))
                * (get_value(m, l, s - Z) + get_value(m, l, s + Z)) * 0.25);
    }

    static MeshEntityId sw(MeshEntityId s, u_int16_t w)
    {
        s.w = w;
        return s;
    }

    template<typename TL, typename TR, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::_cross, TL, TR>>
    get_value(mesh_type const &m, declare::Expression<tags::_cross, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence <I, I>)
    {
        return get_value(m, std::get<0>(expr.m_args_), sw(s, (s.w + 1) % 3)) *
               get_value(m, std::get<1>(expr.m_args_), sw(s, (s.w + 2) % 3)) -
               get_value(m, std::get<0>(expr.m_args_), sw(s, (s.w + 2) % 3)) *
               get_value(m, std::get<1>(expr.m_args_), sw(s, (s.w + 1) % 3));
    }

    template<typename TL, typename TR, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::_dot, TL, TR>>
    get_value(mesh_type const &m, declare::Expression<tags::_dot, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence <I, I>)
    {
        return get_value(m, std::get<0>(expr.m_args_), sw(s, 0)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 0)) +
               get_value(m, std::get<0>(expr.m_args_), sw(s, 1)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 1)) +
               get_value(m, std::get<0>(expr.m_args_), sw(s, 2)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 2));
    }


    template<typename TL, typename TR, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::divides, TL, TR>>
    get_value(mesh_type const &m, declare::Expression<tags::divides, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence <I, VERTEX>)
    {
        return get_value(m, std::get<0>(expr.m_args_), s) /
               _map_to(m, std::get<1>(expr.m_args_), s, index_sequence<VERTEX, I>());
    }

    template<typename TL, typename TR, size_t I>
    static inline traits::value_type_t <declare::Expression<tags::multiplies, TL, TR>>
    get_value(mesh_type const &m, declare::Expression<tags::multiplies, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence <I, VERTEX>
    )
    {
        return get_value(m, std::get<0>(expr.m_args_), s) *
               _map_to(m, std::get<1>(expr.m_args_), s, index_sequence<VERTEX, I>());
    }

    //******************************************************************************************************************
    // for element-wise arithmetic operation
    template<typename TOP, typename ...T, size_t ... I> inline static
    traits::value_type_t <declare::Expression<TOP, T...>>
    _invoke_helper(mesh_type const &m, declare::Expression<TOP, T...> const &expr, MeshEntityId const &s,
                   index_sequence<I...>)
    {
        return expr.m_op_(get_value(m, std::get<I>(expr.m_args_), s)...);
    };


    template<typename TOP, typename ... Args>
    inline static traits::value_type_t <declare::Expression<TOP, Args...>>
    get_value(mesh_type const &m, declare::Expression<TOP, Args...> const &expr, MeshEntityId const &s)
    {
        return _invoke_helper(m, expr, s, make_index_sequence<sizeof...(Args)>());
    }


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T>
    inline static T const &
    get_value(mesh_type const &m, T const &v, MeshEntityId const &s,
              ENABLE_IF(std::is_arithmetic<T>::value)) { return v; }


//    template<typename T>
//    inline static traits::primary_type_t<T>
//    get_value(mesh_type const &m, T const &v, MeshEntityId const &s, ENABLE_IF(traits::is_nTuple<T>::value))
//    {
////        traits::primary_type_t<T> res;
//        res = v;
//        return std::move(res);
//    }
//
//    template<typename TOP, typename ... T>
//    inline static traits::value_type_t<declare::Expression<TOP, T...> >
//    get_value(mesh_type const &m, declare::Expression<TOP, T...> const &expr,
//              MeshEntityId const &s)
//    {
//        return get_value(m, expr, s, traits::iform_list_t<T...>());
//    }




    template<typename TFun> static TV
    get_value(mesh_type const &m, TFun const &fun, MeshEntityId const &s,
    ENABLE_IF((st::is_callable<
                      TFun(point_type const &)>::value))
    )
    {
        return InterpolatePolicy<mesh_type>::template sample<IFORM>(m, s, fun(m.point(s)));
    }

    template<typename TFun> static TV
    get_value(mesh_type const &m, TFun const &fun, MeshEntityId const &s,
              ENABLE_IF((st::is_callable<TFun(mesh_id_type const &), TV>::value)))
    {
        return fun(s);
    }
    //******************************************************************************************************************


    template<typename TOP, typename ...Args> static void
    apply(self_type &self, mesh_type const &m, TOP const &op, Args &&...args)
    {
//        CHECK(m.mesh_block()->range(IFORM, SP_ES_ALL).size());
        m.mesh_block()->range(IFORM, SP_ES_ALL).foreach(
                [&](MeshEntityId const &s)
                {
                    op(get_value(m, self, s), get_value(m, std::forward<Args>(args), s)...);
                });
    }


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

template<typename TV, typename TM, size_type IFORM, size_type DOF>
constexpr Real calculator<declare::Field_<TV, TM, IFORM, DOF> >::m_p_curl_factor_[3];
}//namespace calculus{
//template<typename TM> static  Real  FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh_as::MeshEntityIdCoder, TM>::entity>>::m_p_curl_factor2_[3];
}}//namespace simpla { namespace algebra { namespace declare


#endif /* FDM_H_ */
