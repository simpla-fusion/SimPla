/**
 * @file MeshEntityId.h
 *
 * @date 2015-3-19
 * @author salmon
 */

#ifndef CORE_MESH_MESH_ENTITY_ID_CODER_H_
#define CORE_MESH_MESH_ENTITY_ID_CODER_H_

#include <stddef.h>
#include <limits>
#include <tuple>
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>
#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/mpl/IteratorBlock.h>
#include <simpla/mpl/range.h>
#include <simpla/mpl/type_traits.h>
#include "MeshCommon.h"
#include "EntityIdRange.h"


namespace simpla { namespace mesh
{

//  \verbatim
//
//   |----------------|----------------|---------------|--------------|------------|
//   ^                ^                ^               ^              ^            ^
//   |                |                |               |              |            |
//global          local_outer      local_inner    local_inner    local_outer     global
// _begin          _begin          _begin           _end           _end          _end
//
//  \endverbatim
/**
 *
 *  signed long is 63bit, unsigned long is 64 bit, add a sign bit
 *  \note
 *  \verbatim
 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on bitwise operation
 * 	    H          m  I           m    J           m K
 *  |--------|--------------|--------------|-------------|
 *  |11111111|00000000000000|00000000000000|0000000000000| <= _MH
 *  |00000000|11111111111111|00000000000000|0000000000000| <= _MI
 *  |00000000|00000000000000|11111111111111|0000000000000| <= _MJ
 *  |00000000|00000000000000|00000000000000|1111111111111| <= _MK
 *
 *                      I/J/K
 *  | INDEX_DIGITS------------------------>|
 *  |  Root------------------->| Leaf ---->|
 *  |11111111111111111111111111|00000000000| <=_MRI
 *  |00000000000000000000000001|00000000000| <=_DI
 *  |00000000000000000000000000|11111111111| <=_MTI
 *  | Page NO.->| Tree Root  ->|
 *  |00000000000|11111111111111|11111111111| <=_MASK
 *  \endverbatim
 *
 *  @comment similar to MOAB::EntityHandle but using different code ruler and more efficienct for FD and SAMR  -- salmon. 2016.5.24
 *  @note different get_mesh should use different 'code and hash ruler'  -- salmon. 2016.5.24
 */



struct MeshEntityIdHasher
{
    int64_t operator()(const MeshEntityId &s) const { return s.v; }

    int64_t hash(const MeshEntityId &s) const { return s.v; }
};

struct MeshIdHashCompare
{
    static constexpr inline bool equal(const MeshEntityId &l, const MeshEntityId &r) { return l.v == r.v; }

    static constexpr inline int64_t hash(const MeshEntityId &s) { return s.v; }

};

#define INT_2_ENTITY_ID(_V_) ( *reinterpret_cast<MeshEntityId const *>(&(_V_)))


constexpr inline bool operator==(MeshEntityId const &first, MeshEntityId const &second) { return first.v == second.v; }

constexpr inline MeshEntityId operator-(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{
            first.w,
            static_cast<u_int16_t >(first.z - second.z),
            static_cast<u_int16_t >(first.y - second.y),
            static_cast<u_int16_t >(first.x - second.x)
    };
}

constexpr inline MeshEntityId operator+(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{
            first.w,
            static_cast<u_int16_t >(first.z + second.z),
            static_cast<u_int16_t >(first.y + second.y),
            static_cast<u_int16_t >(first.x + second.x)
    };
}

constexpr inline MeshEntityId operator|(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{.v=  first.v | second.v};
}

constexpr inline bool operator<(MeshEntityId const &first, MeshEntityId const &second)
{
    return first.v < second.v;
}

template<int LEVEL = 4>
struct MeshEntityIdCoder_
{
    /// @name at_level independent
    /// @{

    static constexpr index_type ZERO = static_cast<index_type>((1UL << 13));
    static constexpr int MAX_NUM_OF_NEIGHBOURS = 12;
    static constexpr int ndims = 3;
    static constexpr int MESH_RESOLUTION = 1;

    typedef MeshEntityIdCoder_ this_type;


    /// @name at_level dependent
    /// @{

    static constexpr Real _R = 0.5;

    static constexpr MeshEntityId _DI{0, 0, 0, 1};
    static constexpr MeshEntityId _DJ{0, 0, 1, 0};
    static constexpr MeshEntityId _DK{0, 1, 0, 0};
    static constexpr MeshEntityId _DA{0, 1, 1, 1};



    /// @}

    static constexpr int m_sub_index_to_id_[4][3] = { //

            {0, 0, 0}, /*VERTEX*/
            {1, 2, 4}, /*EDGE*/
            {6, 5, 3}, /*FACE*/
            {7, 7, 7} /*VOLUME*/

    };

    static constexpr int m_id_to_sub_index_[8] = { //

            0, // 000
            0, // 001
            1, // 010
            2, // 011
            2, // 100
            1, // 101
            0, // 110
            0, // 111
    };

    static constexpr MeshEntityId m_id_to_shift_[] = {

            {0, 0, 0, 0},         // 000
            {0, 0, 0, 1},         // 001
            {0, 0, 1, 0},         // 010
            {0, 0, 1, 1},         // 011
            {0, 1, 0, 0},         // 100
            {0, 1, 0, 1},         // 101
            {0, 1, 1, 0},         // 110
            {0, 1, 1, 1},         // 111


    };

    static constexpr Real m_id_to_coordinates_shift_[8][3] = {
            {0.0, 0.0, 0.0},            // 000
            {_R, 0.0, 0.0},           // 001
            {0.0, _R, 0.0},           // 010
            {0.0, 0.0, _R},           // 011
            {_R, _R, 0.0},          // 100
            {_R, 0.0, _R},          // 101
            {0.0, _R, _R},          // 110
            {0.0, _R, _R},          // 111

    };
    static constexpr int m_iform_to_num_of_ele_in_cell_[8] = {
            1, // VETEX
            3, // EDGE
            3, // FACE
            1  // VOLUME
    };
    static constexpr int m_id_to_num_of_ele_in_cell_[] = {

            1,        // 000
            3,        // 001
            3,        // 010
            3,        // 011
            3,        // 100
            3,        // 101
            3,        // 110
            1        // 111
    };

    static constexpr int m_id_to_iform_[] = { //

            0, // 000
            1, // 001
            1, // 010
            2, // 011
            1, // 100
            2, // 101
            2, // 110
            3 // 111
    };

    static MeshEntityId sx(MeshEntityId s, int w)
    {
        s.x = static_cast<u_int16_t>(w);
        return s;
    }

    static MeshEntityId sy(MeshEntityId s, int w)
    {
        s.y = static_cast<u_int16_t>(w);
        return s;
    }

    static MeshEntityId sz(MeshEntityId s, int w)
    {
        s.z = static_cast<u_int16_t>(w);
        return s;
    }

    static MeshEntityId sw(MeshEntityId s, int w)
    {
        s.w = static_cast<u_int16_t>(w);
        return s;
    }

    static constexpr MeshEntityId minimal_vertex(MeshEntityId s)
    {
        return MeshEntityId{.v=s.v & (~_DA.v)};
    }

    template<int IFORM>
    static constexpr int sub_index_to_id(int n = 0)
    {
        return m_sub_index_to_id_[IFORM][n];
    }

    static constexpr int iform(MeshEntityId s)
    {
        return m_id_to_iform_[node_id(s)];
    }

    static constexpr MeshEntityId pack(index_type i0, index_type i1, index_type i2, index_type w = 0)
    {
        return MeshEntityId{static_cast<u_int16_t>(w),
                static_cast<u_int16_t>(i2),
                static_cast<u_int16_t>(i1),
                static_cast<u_int16_t>(i0)};
    }

    template<typename T>
    static constexpr MeshEntityId pack(T const &idx, index_type w = 0)
    {
        return pack(idx[0], idx[1], idx[2], w);
    }

    template<typename T>
    static constexpr MeshEntityId pack_index(T const &idx, index_type n_id = 0)
    {

        return pack_index4(idx[0], idx[1], idx[2], n_id);
    }

    static constexpr MeshEntityId
    pack_index(index_type i, index_type j, index_type k, index_type n_id = 0, index_type w = 0)
    {
        return pack((i + ZERO) << 1, (j + ZERO) << 1, (k + ZERO) << 1, w) | m_id_to_shift_[n_id];
    }

    template<size_t IFORM>
    static constexpr MeshEntityId
    pack_index4(index_type i, index_type j, index_type k, index_type n_id = 0, index_type w = 0)
    {
        return pack((i + ZERO) << 1, (j + ZERO) << 1, (k + ZERO) << 1, w) |
               m_id_to_shift_[m_sub_index_to_id_[IFORM][n_id]];
    }

    static constexpr index_tuple unpack_index(MeshEntityId const &s)
    {
        return index_tuple{
                static_cast<index_type>(s.x >> 1 ) - ZERO,
                static_cast<index_type>(s.y >> 1 ) - ZERO,
                static_cast<index_type>(s.z >> 1 ) - ZERO
        };
    }

    static constexpr nTuple<index_type, 4> unpack_index4(MeshEntityId const &s, index_type dof = 1)
    {
        return nTuple<index_type, 4> {
                static_cast<index_type>(s.x >> 1 ) - ZERO,
                static_cast<index_type>(s.y >> 1 ) - ZERO,
                static_cast<index_type>(s.z >> 1 ) - ZERO,
                static_cast<index_type>(  m_id_to_sub_index_[node_id(s)] * dof + s.w     )
        };
    }

    static constexpr nTuple<index_type, 4> unpack_index4_nodeid(MeshEntityId const &s, index_type dof = 1)
    {
        return nTuple<index_type, 4> {
                static_cast<index_type>(s.x >> 1 ) - ZERO,
                static_cast<index_type>(s.y >> 1 ) - ZERO,
                static_cast<index_type>(s.z >> 1 ) - ZERO,
                static_cast<index_type>( node_id(s) * dof + s.w)
        };
    }
//    template<typename T>
//    static constexpr T type_cast(MeshEntityId s)
//    {
//        return static_cast<T>(unpack(s));
//    }



    static constexpr int num_of_ele_in_cell(MeshEntityId s)
    {
        return m_id_to_num_of_ele_in_cell_[node_id(s)];
    }


    static point_type point(MeshEntityId const &s)
    {
        return point_type{
                static_cast<Real>(s.x - ZERO * 2) * _R,
                static_cast<Real>(s.y - ZERO * 2) * _R,
                static_cast<Real>(s.z - ZERO * 2) * _R
        };
    }

    static std::tuple<MeshEntityId, point_type> point_global_to_local(point_type const &x, int n_id = 0)
    {
        index_tuple i = (x - m_id_to_coordinates_shift_[n_id]) * 2;

        MeshEntityId s = pack(i) | m_id_to_shift_[n_id];

        point_type r = (x - point(s)) / (_R * 2.0);

        return std::make_tuple(s, r);

    }

    static point_type point_local_to_global(MeshEntityId s, point_type const &x) { return point(s) + x * _R * 2; }

    static point_type point_local_to_global(std::tuple<MeshEntityId, point_type> const &t)
    {
        return point_local_to_global(std::get<0>(t), std::get<1>(t));
    }

//! @name id auxiliary functions
//! @{
    static constexpr MeshEntityId m_num_to_id_[] = { //
            {0, 0, 0, 1},
            {0, 0, 1, 0},
            {0, 1, 0, 0}
    };

    static constexpr MeshEntityId DI(int n)
    {
        return m_num_to_id_[n];
    }

    static constexpr MeshEntityId DI(int n, MeshEntityId s)
    {
        return MeshEntityId{.v=s.v & m_num_to_id_[n].v};
    }

    static constexpr MeshEntityId dual(MeshEntityId s)
    {
        return MeshEntityId{.v=(s.v & (~_DA.v)) | ((~(s.v & _DA.v)) & _DA.v)};

    }

    static constexpr MeshEntityId delta_index(MeshEntityId s)
    {
        return MeshEntityId{.v=static_cast<int64_t >(s.v & _DA.v)};
    }

    static constexpr MeshEntityId rotate(MeshEntityId const &s)
    {
        return MeshEntityId{
                static_cast<u_int16_t >(s.w),
                static_cast<u_int16_t >((s.z & ~0x1) | (s.y & 0x1)),
                static_cast<u_int16_t >((s.y & ~0x1) | (s.x & 0x1)),
                static_cast<u_int16_t >((s.x & ~0x1) | (s.z & 0x1))

        };
    }

    static constexpr MeshEntityId inverse_rotate(MeshEntityId const &s)
    {
        return MeshEntityId{
                static_cast<u_int16_t >(s.w),
                static_cast<u_int16_t >((s.z & ~0x1) | (s.x & 0x1)),
                static_cast<u_int16_t >((s.y & ~0x1) | (s.z & 0x1)),
                static_cast<u_int16_t >((s.x & ~0x1) | (s.y & 0x1))
        };
    }

    /**
     *\verbatim
     *                ^y
     *               /
     *        z     /
     *        ^    /
     *    PIXEL0 110-------------111 VOXEL
     *        |  /|              /|
     *        | / |             / |
     *        |/  |    PIXEL1  /  |
     * EDGE2 100--|----------101  |
     *        | m |           |   |
     *        |  010----------|--011 PIXEL2
     *        |  / EDGE1      |  /
     *        | /             | /
     *        |/              |/
     *       000-------------001---> x
     *                       EDGE0
     *
     *\endverbatim
     */

    static constexpr int NUM_OF_NODE_ID = 8;
    enum node_id_tag
    {
        TAG_VERTEX = 0,
        TAG_EDGE0 = 1,
        TAG_EDGE1 = 2,
        TAG_EDGE2 = 4,
        TAG_FACE0 = 6,
        TAG_FACE1 = 5,
        TAG_FACE2 = 3,
        TAG_VOLUME = 7
    };

    static constexpr int node_id(MeshEntityId const &s)
    {
        return (s.x & 0x1) | ((s.y & 0x1) << 1) | ((s.z & 0x1) << 2);
    }

    static constexpr int m_id_to_index_[8] = { //

            0, // 000
            0, // 001
            1, // 010
            2, // 011
            2, // 100
            1, // 101
            0, // 110
            0, // 111
    };

    static constexpr int sub_index(MeshEntityId const &s)
    {
        return m_id_to_index_[node_id(s)];
    }

    /**
     * \verbatim
     *                ^y
     *               /
     *        z     /
     *        ^
     *        |   6---------------7
     *        |  /|              /|
     *          / |             / |
     *         /  |            /  |
     *        4---|-----------5   |
     *        |   |     x0    |   |
     *        |   2-----------|---3
     *        |  /            |  /
     *        | /             | /
     *        |/              |/
     *        0---------------1   ---> x
     *
     *   \endverbatim
     */
    static constexpr int MAX_NUM_OF_ADJACENT_CELL = 12;

    static constexpr int m_adjacent_cell_num_[4/* to iform*/][8/* node id*/] =

            { // VERTEX
                    {
                            /* 000*/1,
                            /* 001*/2,
                            /* 010*/2,
                            /* 011*/4,
                            /* 100*/2,
                            /* 101*/4,
                            /* 110*/4,
                            /* 111*/8
                    },

                    // EDGE
                    {
                            /* 000*/6,
                            /* 001*/1,
                            /* 010*/1,
                            /* 011*/4,
                            /* 100*/1,
                            /* 101*/4,
                            /* 110*/4,
                            /* 111*/12
                    },

                    // FACE
                    {
                            /* 000*/12,
                            /* 001*/4,
                            /* 010*/4,
                            /* 011*/1,
                            /* 100*/4,
                            /* 101*/1,
                            /* 110*/1,
                            /* 111*/6
                    },

                    // VOLUME
                    {
                            /* 000*/8,
                            /* 001*/4,
                            /* 010*/4,
                            /* 011*/2,
                            /* 100*/4,
                            /* 101*/2,
                            /* 110*/2,
                            /* 111*/1
                    }

            };

    static constexpr MeshEntityId
            m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/] =
            {
                    //To VERTEX
                    {

                            /* 000*/
                            {//
                                    _DA
                            },
                            /* 001*/
                            {       //
                                    _DA - _DI,
                                    _DA + _DI
                            },
                            /* 010*/
                            {       //
                                    _DA - _DJ,
                                    _DA + _DJ
                            },
                            /* 011*/
                            {//
                                    _DA - _DI - _DJ, /* 000*/
                                    _DA + _DI - _DJ, /* 001*/
                                    _DA - _DI + _DJ, /* 010*/
                                    _DA + _DI + _DJ /* 011 */
                            },
                            /* 100*/
                            {//
                                    _DA - _DK,
                                    _DA + _DK
                            },
                            /* 101*/
                            {       //
                                    _DA - _DK - _DI, /*000*/
                                    _DA - _DK + _DI, /*001*/
                                    _DA + _DK - _DI, /*100*/
                                    _DA + _DK + _DI /*101*/
                            },
                            /* 110*/
                            {//
                                    _DA - _DJ - _DK, /*000*/
                                    _DA + _DJ - _DK, /*010*/
                                    _DA - _DJ + _DK, /*100*/
                                    _DA + _DJ + _DK /*110*/
                            },
                            /* 111*/
                            {       //
                                    _DA - _DK - _DJ - _DI, /*000*/
                                    _DA - _DK - _DJ + _DI, /*001*/
                                    _DA - _DK + _DJ - _DI, /*010*/
                                    _DA - _DK + _DJ + _DI, /*011*/
                                    _DA + _DK - _DJ - _DI, /*100*/
                                    _DA + _DK - _DJ + _DI, /*101*/
                                    _DA + _DK + _DJ - _DI, /*110*/
                                    _DA + _DK + _DJ + _DI  /*111*/

                            }

                    },

                    //To EDGE
                    {
                            /* 000*/
                            {       //
                                    _DA + _DI,
                                    _DA - _DI,
                                    _DA + _DJ,
                                    _DA - _DJ,
                                    _DA + _DK,
                                    _DA - _DK
                            },
                            /* 001*/
                            {
                                    _DA
                            },
                            /* 010*/
                            {
                                    _DA
                            },
                            /* 011*/
                            {        //
                                    _DA - _DJ,
                                    _DA + _DI,
                                    _DA + _DJ,
                                    _DA - _DI
                            },
                            /* 100*/
                            {       //
                                    _DA
                            },
                            /* 101*/
                            {         //
                                    _DA - _DI,
                                    _DA + _DK,
                                    _DA + _DI,
                                    _DA - _DK
                            },
                            /* 110*/
                            {       //
                                    _DA - _DK,
                                    _DA + _DJ,
                                    _DA + _DK,
                                    _DA - _DJ
                            },
                            /* 111*/
                            {       //
                                    _DA - _DK - _DJ,  //-> 001
                                    _DA - _DK + _DI,  //   012
                                    _DA - _DK + _DJ,  //   021
                                    _DA - _DK - _DI,  //   010

                                    _DA - _DI - _DJ,  //
                                    _DA - _DI + _DJ,  //
                                    _DA + _DI - _DJ,  //
                                    _DA + _DI + _DJ,  //

                                    _DA + _DK - _DJ,  //
                                    _DA + _DK + _DI,  //
                                    _DA + _DK + _DJ,  //
                                    _DA + _DK - _DI  //
                            }},

                    //To FACE
                    {
                            /* 000*/
                            {       //
                                    _DA - _DK - _DJ,  //
                                    _DA - _DK + _DI,  //
                                    _DA - _DK + _DJ,  //
                                    _DA - _DK - _DI,  //

                                    _DA - _DI - _DJ,  //
                                    _DA - _DI + _DJ,  //
                                    _DA + _DI - _DJ,  //
                                    _DA + _DI + _DJ,  //

                                    _DA + _DK - _DJ,  //
                                    _DA + _DK + _DI,  //
                                    _DA + _DK + _DJ,  //
                                    _DA + _DK - _DI  //
                            },
                            /* 001*/
                            {       //
                                    _DA - _DJ,          //
                                    _DA + _DK,   //
                                    _DA + _DJ,   //
                                    _DA - _DK    //
                            },
                            /* 010*/
                            {       //
                                    _DA - _DK,          //
                                    _DA + _DI,   //
                                    _DA + _DK,   //
                                    _DA - _DI    //
                            },
                            /* 011*/
                            {_DA},
                            /* 100*/
                            {//
                                    _DA - _DI,         //
                                    _DA + _DJ,  //
                                    _DA + _DI,  //
                                    _DA - _DJ   //
                            },
                            /* 101*/
                            {       //
                                    _DA
                            },
                            /* 110*/
                            {       //
                                    _DA
                            },
                            /* 111*/
                            {       //
                                    _DA - _DI,         //
                                    _DA - _DJ,  //
                                    _DA - _DK,  //
                                    _DA + _DI,  //
                                    _DA + _DJ,  //
                                    _DA + _DK   //
                            }},
                    // TO VOLUME
                    {
                            /* 000*/
                            {       //
                                    _DA - _DI - _DJ - _DK,  //
                                    _DA - _DI + _DJ - _DK,  //
                                    _DA - _DI - _DJ + _DK,  //
                                    _DA - _DI + _DJ + _DK,  //

                                    _DA + _DI - _DJ - _DK,  //
                                    _DA + _DI + _DJ - _DK,  //
                                    _DA + _DI - _DJ + _DK,  //
                                    _DA + _DI + _DJ + _DK  //

                            },
                            /* 001*/
                            {       //
                                    _DA - _DJ - _DK,           //
                                    _DA - _DJ + _DK,    //
                                    _DA + _DJ - _DK,    //
                                    _DA + _DJ + _DK     //
                            },
                            /* 010*/
                            {        //
                                    _DA - _DK - _DI,  //
                                    _DA - _DK + _DI,  //
                                    _DA + _DK - _DI,  //
                                    _DA + _DK + _DI   //
                            },
                            /* 011*/
                            {       //
                                    _DA - _DK,
                                    _DA + _DK},
                            /* 100*/
                            {         //
                                    _DA - _DI - _DJ,   //
                                    _DA - _DI + _DJ,   //
                                    _DA + _DI - _DJ,   //
                                    _DA + _DI + _DJ    //
                            },
                            /* 101*/
                            {//
                                    _DA - _DJ,
                                    _DA + _DJ
                            },
                            /* 110*/
                            {       //
                                    _DA - _DI,
                                    _DA + _DI
                            },
                            /* 111*/
                            {//
                                    _DA
                            }
                    }

            };

    static int get_adjacent_entities(int IFORM, MeshEntityId s, MeshEntityId *res = nullptr)
    {
        return get_adjacent_entities(IFORM, node_id(s), s, res);
    }

    static int get_adjacent_entities(int IFORM, int nodeid, MeshEntityId s, MeshEntityId *res = nullptr)
    {
        if (res != nullptr)
        {
            for (int i = 0; i < m_adjacent_cell_num_[IFORM][nodeid]; ++i)
            {
                res[i] = s - _DA + m_adjacent_cell_matrix_[IFORM][nodeid][i];
            }
        }
        return m_adjacent_cell_num_[IFORM][nodeid];
    }

    struct range_type
    {
    private:
        typedef range_type this_type;
    public:


        range_type()
                : m_iform_(VERTEX), m_min_(), m_max_(m_min_), m_grain_size_(m_min_), m_dof_(1) {}

        // constructors

        range_type(index_tuple const &b, index_tuple const &e, MeshEntityType IFORM = VERTEX, index_type dof = 1)
                : m_iform_(IFORM), m_min_(b), m_max_(e), m_dof_(dof)
        {
            m_grain_size_ = 1;
            for (int i = 0; i < ndims; ++i)
            {
                if (m_max_[i] - m_min_[i] <= m_grain_size_[i])
                {
                    m_grain_size_[i] = m_max_[i] - m_min_[i];
                }
            }
        }

        range_type(this_type const &r)
                : m_iform_(r.m_iform_), m_min_(r.m_min_), m_max_(r.m_max_), m_grain_size_(r.m_grain_size_),
                  m_dof_(r.m_dof_)
        {
        }

        range_type(index_tuple const &b, index_tuple const &e, index_tuple const &grain_size,
                   MeshEntityType IFORM = VERTEX, index_type dof = 1)
                : m_iform_(IFORM), m_min_(b), m_max_(e), m_grain_size_(grain_size), m_dof_(dof)
        {
        }

        range_type(range_type &r, tags::split)
                : m_iform_(r.m_iform_), m_min_(r.m_min_), m_max_(r.m_max_), m_grain_size_(r.m_grain_size_),
                  m_dof_(r.m_dof_)
        {

            ASSERT(is_divisible());

            int n = 0;

            index_type L = 0;

            for (int i = 0; i < ndims; ++i)
            {
                if ((m_max_[i] - m_min_[i] > L) && (m_max_[i] - m_min_[i] > m_grain_size_[i]))
                {
                    n = i;
                    L = m_max_[i] - m_min_[i];
                }
            }
            m_max_[n] = m_min_[n] + L / 2;
            r.m_min_[n] = m_max_[n];
        }

        range_type(this_type &r, tags::proportional_split const &proportion)
        {
            int n = 0;
            index_type L = m_max_[0] - m_min_[0];
            for (int i = 1; i < ndims; ++i)
            {
                if (m_max_[i] - m_min_[i] > L)
                {
                    n = i;
                    L = m_max_[i] - m_min_[i];
                }
            }

            m_max_[n] = m_min_[n] + L * proportion.left() /
                                    ((proportion.left() + proportion.right() > 0) ? (proportion.left() +
                                                                                     proportion.right()) : 1);
            r.m_min_[n] = m_max_[n];
        }

        ~range_type() {}

        void swap(this_type &other)
        {
            std::swap(m_iform_, other.m_iform_);
            std::swap(m_dof_, other.m_dof_);

            std::swap(m_min_, other.m_min_);
            std::swap(m_max_, other.m_max_);
            std::swap(m_grain_size_, other.m_grain_size_);
        }

        MeshEntityType entity_type() const { return m_iform_; }

        index_box_type index_box() const { return std::make_tuple(m_min_, m_max_); }

        // Proportional split is enabled
        static const bool is_splittable_in_proportion = true;

        // capacity


        bool empty() const { return m_min_ == m_max_; }

        size_t size() const
        {
            return static_cast<size_t>(((m_iform_ == VERTEX || m_iform_ == VOLUME) ? 1 : 3)
                                       * (m_max_[0] - m_min_[0]) * (m_max_[1] - m_min_[1]) * (m_max_[2] - m_min_[2]));
        }

        // access
        index_tuple const &grainsize() const { return m_grain_size_; }

        bool is_divisible() const
        {
            int count = 0;

            for (int i = 0; i < ndims; ++i)
            {
                if (m_max_[i] - m_min_[i] <= m_grain_size_[i]) { ++count; }
            }
            return count < ndims;
        }

        template<typename Body>
        void foreach(Body const &body) const
        {
            range_type const &r = *this;
            int ib = r.m_min_[0];
            int ie = r.m_max_[0];
#pragma omp parallel for
            for (int i = ib; i < ie; ++i)
            {
                for (index_type j = r.m_min_[1], je = r.m_max_[1]; j < je; ++j)
                    for (index_type k = r.m_min_[2], ke = r.m_max_[2]; k < ke; ++k)
                        for (index_type n = 0, ne = m_iform_to_num_of_ele_in_cell_[r.m_iform_]; n < ne; ++n)
                            for (index_type w = 0; w < m_dof_; ++w)
                            {
                                body(pack_index(i, j, k, m_sub_index_to_id_[r.m_iform_][n], w));
                            }
            }
        }

    private:
        index_type m_dof_ = 1;
        size_type m_iform_;
        index_tuple m_min_, m_max_, m_grain_size_;
    };

//    typedef RangeHolder<iterator> range_type;

    static range_type make_range(index_tuple const &min, index_tuple const &max, size_type iform = VERTEX)
    {
        return range_type(min, max, iform);
    }

    static range_type make_range(index_box_type const &b, size_type iform = VERTEX)
    {
        return make_range(std::get<0>(b), std::get<1>(b), iform);
    }


    static size_type
    hash(index_type i, index_type j, index_type k, int nid, index_tuple const &b, index_tuple const &e)
    {
        //C-ORDER SLOW FIRST
        return
                static_cast<size_type>(
                        ((k + e[2] - b[2] - b[2]) % (e[2] - b[2]) +
                         (((j + e[1] - b[1] - b[1]) % (e[1] - b[1])) +
                          ((i + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])) * (e[2] - b[2])
                        ) * m_id_to_num_of_ele_in_cell_[nid] + m_id_to_index_[nid]);

    }

    static size_type hash(MeshEntityId const &s, index_tuple const &b, index_tuple const &e)
    {
        //C-ORDER SLOW FIRST

        return hash(s.x >> 1, s.y >> 1, s.z >> 1, node_id(s), b, e);
//                (
//                        ((s.z >> 1) + e[2] - b[2] - b[2]) % (e[2] - b[2]) +
//
//                        (
//                                (((s.y >> 1) + e[1] - b[1] - b[1]) % (e[1] - b[1])) +
//
//                                (((s.x >> 1) + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])
//
//                        ) * (e[2] - b[2])
//
//                ) * num_of_ele_in_cell(s) + sub_index(s);

    }

    static index_type hash2(MeshEntityId const &s, index_tuple const &b, size_tuple const &l)
    {
        //C-ORDER SLOW FIRST

        return
                (
                        ((s.z >> 1) + l[2] - b[2]) % (l[2]) +

                        (
                                (((s.y >> 1) + l[1] - b[1]) % (l[1])) +

                                (((s.x >> 1) + l[0] - b[0]) % (l[0])) * (l[1])

                        ) * (l[2])

                ) * num_of_ele_in_cell(s) + sub_index(s);

    }

    template<int IFORM>
    static constexpr size_t max_hash(MeshEntityId b, MeshEntityId e)
    {
        return max_hash(unpack_index(e), unpack_index(b), IFORM);
    }

    static constexpr size_t max_hash(index_tuple const &b, index_tuple const &e, size_type IFORM)
    {
        return static_cast<size_t>((e[2] - b[2]) * (e[1] - b[1]) * (e[0] - b[0])
                                   * m_id_to_num_of_ele_in_cell_[m_sub_index_to_id_[IFORM][0]]);
    }

};

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<int L> constexpr int MeshEntityIdCoder_<L>::ndims;

template<int L> constexpr int MeshEntityIdCoder_<L>::MESH_RESOLUTION;

template<int L> constexpr Real MeshEntityIdCoder_<L>::_R;

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::_DK;

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::_DJ;

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::_DI;

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::_DA;

template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_index_[];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_iform_[];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_num_of_ele_in_cell_[];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_adjacent_cell_num_[4][8];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_iform_to_num_of_ele_in_cell_[];

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::m_num_to_id_[];

template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::m_id_to_shift_[];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_sub_index_[];

template<int L> constexpr int MeshEntityIdCoder_<L>::m_sub_index_to_id_[4][3];

template<int L> constexpr MeshEntityId
        MeshEntityIdCoder_<L>::m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][
        MAX_NUM_OF_ADJACENT_CELL/*id shift*/];

template<int L> constexpr Real MeshEntityIdCoder_<L>::m_id_to_coordinates_shift_[8][3];

typedef MeshEntityIdCoder_<> MeshEntityIdCoder;

}//namespace  get_mesh
}// namespace simpla

#endif /* CORE_MESH_MESH_ENTITY_ID_CODER_H_ */

