//
// Created by salmon on 17-5-29.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include "simpla/SIMPLA_config.h"

#include "simpla/data/SPObject.h"
#include "simpla/geometry/GeoObject.h"
#include "simpla/utilities/Signal.h"

namespace simpla {
namespace geometry {
class Curve;
struct Chart : public SPObject {
    SP_OBJECT_HEAD(Chart, SPObject)

   private:
    bool m_is_setup_ = false;

   public:
    virtual void SetUp() { m_is_setup_ = true; }
    virtual bool isSetUp() { return true; }
    virtual void Update() { ASSERT(isSetUp()); };
    virtual void TearDown() { m_is_setup_ = false; };

    virtual std::shared_ptr<Curve> GetAxis(point_type const &x0, int dir, Real length) const { return nullptr; };
    virtual std::shared_ptr<Curve> GetAxis(index_tuple const &x0, int dir, index_type length) const { return nullptr; };

    virtual int GetNDIMS() const;
    void SetLevel(int level);
    int GetLevel() const;

    void SetOrigin(point_type const &x);
    point_type const &GetOrigin() const;

    void SetScale(point_type const &x);
    point_type const &GetScale() const;

    point_type GetCellWidth(int level = 0) const;

    template <typename... Args>
    point_type uvw(Args &&... args) const {
        return local_coordinates(0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    point_type xyz(Args &&... args) const {
        return global_coordinates(std::forward<Args>(args)...);
    }
    template <typename TR>
    point_type local_coordinates(TR const &x) const {
        return m_axis_.xyz(x);
    }

    point_type local_coordinates(std::tuple<point_type, index_tuple> const &r) const {
        return local_coordinates(&std::get<0>(r)[0], std::get<1>(r));
    }
    point_type local_coordinates(point_type const &r, index_tuple const &x) const {
        return local_coordinates(&r[0], x);
    }
    point_type local_coordinates(Real const *r, index_tuple const &x) const {
        return local_coordinates(point_type{x[0] + r[0], x[1] + r[1], x[2] + r[2]});
    }
    static constexpr Real m_id_to_coordinates_shift_[8][3] = {
        {0.0, 0.0, 0.0},  // 000
        {0.5, 0.0, 0.0},  // 001
        {0.0, 0.5, 0.0},  // 010
        {0.5, 0.5, 0.0},  // 011
        {0.0, 0.0, 0.5},  // 100
        {0.5, 0.0, 0.5},  // 101
        {0.0, 0.5, 0.5},  // 110
        {0.5, 0.5, 0.5},  // 111

    };
    point_type local_coordinates(int tag, index_tuple const &x) const {
        return local_coordinates(m_id_to_coordinates_shift_[tag], x);
    }
    point_type local_coordinates(int tag, index_type x, index_type y, index_type z) const {
        return local_coordinates(m_id_to_coordinates_shift_[tag], x, y, z);
    }

    template <typename TR>
    point_type local_coordinates(TR const &r, index_type x, index_type y, index_type z) const {
        return local_coordinates(point_type{x + r[0], y + r[1], z + r[2]});
    }

    template <typename TR>
    std::tuple<point_type, index_tuple> invert_local_coordinates(TR const &x) const {
        // NOTE: require 0 < r < 1- epsilon
        static constexpr Real epsilon = 1.0e-8;
        point_type r = m_axis_.uvw(x);
        index_tuple id{0, 0, 0};

        id[0] = static_cast<index_type>(floor(r[0]));
        id[1] = static_cast<index_type>(floor(r[1]));
        id[2] = static_cast<index_type>(floor(r[2]));

        r[0] = std::fdim(r[0] - epsilon, id[0]);
        r[1] = std::fdim(r[1] - epsilon, id[1]);
        r[2] = std::fdim(r[2] - epsilon, id[2]);

        return std::make_tuple(r, id);
    }

    template <typename TR>
    std::tuple<point_type, index_tuple> invert_global_coordinates(TR const &x) const {
        return invert_local_coordinates(inv_map(x));
    }

    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    };

    virtual point_type map(point_type const &x) const { return x; }

    virtual point_type inv_map(point_type const &x) const { return x; }

    virtual Real length(point_type const &p0, point_type const &p1) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, point_type const &p2) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1, point_type const &p2,
                        point_type const &p3) const = 0;

    virtual Real length(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1) const = 0;

    virtual Real inner_product(point_type const &uvw, vector_type const &v0, vector_type const &v1) const = 0;

   private:
    template <size_t... I, typename PointS>
    auto _volume(std::integer_sequence<size_t, I...> _, PointS const &points) const {
        return volume(map(std::get<I>(points))...);
    }

   public:
    template <typename... P>
    auto volume(std::tuple<P...> const &points) const {
        return _volume(std::index_sequence_for<P...>(), points);
    }

   private:
    template <size_t... I, typename PointS>
    auto _MapToBase(std::integer_sequence<size_t, I...> _, PointS const &points) const {
        return std::make_tuple(map(std::get<I>(points))...);
    }
    template <size_t... I, typename PointS>
    auto _InvMapFromBase(std::integer_sequence<size_t, I...> _, PointS const &points) const {
        return std::make_tuple(inv_map(std::get<I>(points))...);
    }

   public:
    template <typename... P>
    auto MapToBase(std::tuple<P...> const &points) const {
        return _MapToBase(std::index_sequence_for<P...>(), points);
    }
    template <typename... P>
    auto InvMapFromBase(std::tuple<P...> const &points) const {
        return _InvMapFromBase(std::index_sequence_for<P...>(), points);
    }

    void SetAxis(Axis const &a) { m_axis_ = a; }
    Axis &GetAxis() { return m_axis_; }
    Axis const &GetAxis() const { return m_axis_; }

   protected:
    int m_level_ = 0;
    Axis m_axis_;
};
}
}
#endif  // SIMPLA_CHART_H
