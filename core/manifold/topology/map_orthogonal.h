/**
 * @file map_orthogonal.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_MAP_ORTHOGONAL_H
#define SIMPLA_MAP_ORTHOGONAL_H

#include "../../gtl/primitives.h"

namespace simpla
{


template<typename T0, typename T1=IdentifyMap, typename T2=IdentifyMap>
struct OrthogonalMap
{
private:

    typedef OrthogonalMap this_type;

    T0 x_map;
    T1 y_map;
    T2 z_map;

public:


    OrthogonalMap() { }

    ~OrthogonalMap() { }

    void swap(this_type &other)
    {
        std::swap(x_map, other.x_map);
        std::swap(y_map, other.y_map);
        std::swap(z_map, other.z_map);
    }

    template<typename TB0, typename TB1, typename TD>
    void set(TB0 const &src_box, TB1 const &dest_box, TD const &dims)
    {

        auto src_min_ = std::get<0>(src_box);
        auto src_max_ = std::get<1>(src_box);

        auto dest_min = std::get<0>(dest_box);
        auto dest_max = std::get<1>(dest_box);


        x_map.set(src_min_[0], src_max_[0], dest_min[0], dest_max[0], dims[0] <= 1);
        y_map.set(src_min_[1], src_max_[1], dest_min[1], dest_max[1], dims[1] <= 1);
        z_map.set(src_min_[2], src_max_[2], dest_min[2], dest_max[2], dims[2] <= 1);
    }

    void deploy()
    {
    }

    template<typename TP>
    TP map(TP const &x) const
    {

//        TP res;
//
//
//        res[0] = x_map.map(x[0]);//std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);
//        res[1] = y_map.map(x[1]);//std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);
//        res[2] = z_map.map(x[2]);//std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);
//
//        return std::move(res);

        return std::move(TP({x_map.map(x[0]), y_map.map(x[1]), z_map.map(x[2])}));
    }

    template<typename TP>
    TP inv_map(TP const &y) const
    {


//        TP res;
//        res[0] = x_map.inv_map(y[0]);//std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);
//        res[1] = y_map.inv_map(y[1]);//std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);
//        res[2] = z_map.inv_map(y[2]);//std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);

        return std::move(TP({x_map.inv_map(y[0]), y_map.inv_map(y[1]), z_map.inv_map(y[2])}));
    }

//
//    Real volume(int n, point_type const x0[], bool is_box = false) const
//    {
//        Real res = 0.0;
//        switch (n)
//        {
//            case 2: // line or box
//                if (is_box)
//                {
//
//                }
//                else
//                {
//
//                }
//                break;
//            case 4: // rectangle
//                break;
//            case 8: // Hexahedron
//                break;
//            default:
//                ERROR("illegal point number!");
//                break;
//        }
//
//        return res;
//    }
//
//    template<typename TB>
//    Real box_volume(TB const &b) const
//    {
//        point_type x0, x1;
//        std::tie(x0, x1) = b;
//    }


};
}//namespace simpla
#endif //SIMPLA_MAP_ORTHOGONAL_H
