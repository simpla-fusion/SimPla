/**
 * @file EntityId.h
 *
 * @date 2015-3-19
 * @author salmon
 */

#ifndef CORE_MESH_MESH_ENTITY_ID_CODER_H_
#define CORE_MESH_MESH_ENTITY_ID_CODER_H_

#include <simpla/utilities/nTuple.h>
#include <stddef.h>
#include <limits>
#include <set>
#include <tuple>
#include "Log.h"
#include "Range.h"
#include "nTupleExt.h"
#include "sp_def.h"
#include "type_traits.h"
namespace simpla {
// typedef union { struct { u_int8_t w, z, y, x; }; int32_t v; } EntityId32;

typedef union {
    struct {
        int16_t x, y, z, w;
    };
    int64_t v;
} EntityId64;
typedef EntityId64 EntityId;

/**
 *  @comment similar to MOAB::EntityHandle but using different code ruler and more efficient for FD
 * and SAMR  -- salmon. 2016.5.24
 *  @note different get_mesh should use different 'code and hash ruler'  -- salmon. 2016.5.24
 */

#define INT_2_ENTITY_ID(_V_) (*reinterpret_cast<EntityId const*>(&(_V_)))

constexpr inline bool operator==(EntityId first, EntityId second) { return first.v == second.v; }

constexpr inline EntityId operator-(EntityId first, EntityId second) {
    return EntityId{.x = static_cast<int16_t>(first.x - second.x),
                    .y = static_cast<int16_t>(first.y - second.y),
                    .z = static_cast<int16_t>(first.z - second.z),
                    .w = first.w};
}

constexpr inline EntityId operator+(EntityId first, EntityId second) {
    return EntityId{.x = static_cast<int16_t>(first.x + second.x),
                    .y = static_cast<int16_t>(first.y + second.y),
                    .z = static_cast<int16_t>(first.z + second.z),
                    .w = first.w};
}

constexpr inline EntityId operator|(EntityId first, EntityId second) { return EntityId{.v = first.v | second.v}; }

constexpr inline bool operator<(EntityId first, EntityId second) { return first.v < second.v; }

struct EntityIdCoder {
    /// @name at_level independent
    /// @{

    static constexpr index_type ZERO = static_cast<index_type>((1UL << 13));
    static constexpr int MAX_NUM_OF_NEIGHBOURS = 12;
    static constexpr int ndims = 3;
    static constexpr int MESH_RESOLUTION = 1;

    static constexpr EntityId _DI{.x = 1, .y = 0, .z = 0, .w = 0};
    static constexpr EntityId _DJ{.x = 0, .y = 1, .z = 0, .w = 0};
    static constexpr EntityId _DK{.x = 0, .y = 0, .z = 1, .w = 0};
    static constexpr EntityId _DA{.x = 1, .y = 1, .z = 1, .w = static_cast<int16_t>(-1)};

    typedef EntityIdCoder this_type;

    /// @name at_level dependent
    /// @{

    static constexpr Real _R = 0.5;

    /// @}

    static constexpr int m_sub_index_to_id_[4][3] = {
        //
        {0, 0, 0}, /*VERTEX*/
        {1, 2, 4}, /*EDGE*/
        {6, 5, 3}, /*FACE*/
        {7, 7, 7}  /*VOLUME*/

    };

    static constexpr int m_id_to_sub_index_[8] = {
        //

        0,  // 000
        0,  // 001
        1,  // 010
        2,  // 011
        2,  // 100
        1,  // 101
        0,  // 110
        0,  // 111
    };

    static constexpr EntityId m_id_to_shift_[] = {
        {.x = 0, .y = 0, .z = 0, .w = 0},  // 000
        {.x = 1, .y = 0, .z = 0, .w = 0},  // 001
        {.x = 0, .y = 1, .z = 0, .w = 0},  // 010
        {.x = 1, .y = 1, .z = 0, .w = 0},  // 011
        {.x = 0, .y = 0, .z = 1, .w = 0},  // 100
        {.x = 1, .y = 0, .z = 1, .w = 0},  // 101
        {.x = 0, .y = 1, .z = 1, .w = 0},  // 110
        {.x = 1, .y = 1, .z = 1, .w = 0},  // 111

    };

    static constexpr Real m_id_to_coordinates_shift_[8][3] = {
        {0.0, 0.0, 0.0},  // 000
        {_R, 0.0, 0.0},   // 001
        {0.0, _R, 0.0},   // 010
        {0.0, 0.0, _R},   // 011
        {_R, _R, 0.0},    // 100
        {_R, 0.0, _R},    // 101
        {0.0, _R, _R},    // 110
        {0.0, _R, _R},    // 111

    };
    static constexpr int m_iform_to_num_of_ele_in_cell_[8] = {
        1,  // VETEX
        3,  // EDGE
        3,  // FACE
        1   // VOLUME
    };
    static constexpr int m_id_to_num_of_ele_in_cell_[] = {

        1,  // 000
        3,  // 001
        3,  // 010
        3,  // 011
        3,  // 100
        3,  // 101
        3,  // 110
        1   // 111
    };

    static constexpr int m_id_to_iform_[] = {
        //

        0,  // 000
        1,  // 001
        1,  // 010
        2,  // 011
        1,  // 100
        2,  // 101
        2,  // 110
        3   // 111
    };

    static EntityId sx(EntityId s, int w) {
        s.x = static_cast<int16_t>(w);
        return s;
    }

    static EntityId sy(EntityId s, int w) {
        s.y = static_cast<int16_t>(w);
        return s;
    }

    static EntityId sz(EntityId s, int w) {
        s.z = static_cast<int16_t>(w);
        return s;
    }

    static EntityId sw(EntityId s, int w) {
        s.w = static_cast<int16_t>(w);
        return s;
    }
    static EntityId tag(EntityId s, int64_t tag) {
        s.v = (s.v & (~_DA.v)) | tag;
        return s;
    }
    static EntityId minimal_vertex(EntityId s) { return EntityId{.v = s.v & (~_DA.v)}; }

    template <int IFORM>
    static constexpr int sub_index_to_id(int n = 0) {
        return m_sub_index_to_id_[IFORM][n];
    }

    static constexpr int iform(EntityId s) { return m_id_to_iform_[node_id(s)]; }

    //    static constexpr EntityId pack(index_type i0, index_type i1, index_type i2, index_type w = 0) {
    //        return EntityId{.w = static_cast<int16_t>(w),
    //                        .x = static_cast<int16_t>(i0),
    //                        .y = static_cast<int16_t>(i1),
    //                        .z = static_cast<int16_t>(i2)};
    //    }
    //
    //    template <typename T>
    //    static constexpr EntityId pack(T const& idx, index_type w = 0) {
    //        return pack(idx[0], idx[1], idx[2], w);
    //    }
    //
    //    template <typename T>
    //    static constexpr EntityId pack_index(T const& idx, index_type n_id = 0) {
    //        return pack_index4(idx[0], idx[1], idx[2], n_id);
    //    }
    //
    //    static constexpr EntityId pack_index(index_type i, index_type j, index_type k, index_type n_id = 0,
    //                                         index_type w = 0) {
    //        return pack((i + ZERO) << 1, (j + ZERO) << 1, (k + ZERO) << 1, w) | m_id_to_shift_[n_id];
    //    }
    //
    //    template <int IFORM>
    //    static constexpr EntityId pack_index4(index_type i, index_type j, index_type k, index_type n_id = 0,
    //                                          index_type w = 0) {
    //        return pack((i + ZERO) << 1, (j + ZERO) << 1, (k + ZERO) << 1, w) |
    //               m_id_to_shift_[m_sub_index_to_id_[IFORM][n_id]];
    //    }
    //
    //    static index_tuple unpack_index(EntityId s) {
    //        return index_tuple{static_cast<index_type>(s.x >> 1) - ZERO, static_cast<index_type>(s.y >> 1) - ZERO,
    //                           static_cast<index_type>(s.z >> 1) - ZERO};
    //    }
    //
    //    static nTuple<index_type, 4> unpack_index4(EntityId s, index_type dof = 1) {
    //        return nTuple<index_type, 4>{static_cast<index_type>(s.x >> 1) - ZERO,
    //                                     static_cast<index_type>(s.y >> 1) - ZERO,
    //                                     static_cast<index_type>(s.z >> 1) - ZERO,
    //                                     static_cast<index_type>(m_id_to_sub_index_[node_id(s)] * dof + s.w)};
    //    }
    //
    //    static nTuple<index_type, 4> unpack_index4_nodeid(EntityId s, index_type dof = 1) {
    //        return nTuple<index_type, 4>{static_cast<index_type>(s.x >> 1) - ZERO, static_cast<index_type>(s.y >> 1) -
    //        ZERO,
    //                                     static_cast<index_type>(s.z >> 1) - ZERO,
    //                                     static_cast<index_type>(node_id(s) * dof + s.w)};
    //    }
    //    template<typename T>
    //    static constexpr T type_cast(EntityId s)
    //    {
    //        return static_cast<T>(unpack(s));
    //    }
    template <int IFORM>
    static EntityId Pack(index_type i, index_type j, index_type k, unsigned int n, unsigned int d) {
        EntityId s;
        s.x = static_cast<int16_t>(i);
        s.y = static_cast<int16_t>(j);
        s.z = static_cast<int16_t>(k);
        s.w = static_cast<int16_t>((d << 3) | m_sub_index_to_id_[IFORM][n]);
        return s;
    }
    template <int IFORM, int DOF>
    static int SubIndex(EntityId s) {
        return m_id_to_sub_index_[s.w & 0x7] * DOF + (s.w >> 3);
    }

    static constexpr int num_of_ele_in_cell(EntityId s) { return m_id_to_num_of_ele_in_cell_[node_id(s)]; }

    //    static point_type point(EntityId s) {
    //        return point_type{static_cast<Real>(s.x - ZERO * 2) * _R,
    //                          static_cast<Real>(s.y - ZERO * 2) * _R,
    //                          static_cast<Real>(s.z - ZERO * 2) * _R};
    //    }
    //
    //    static std::tuple<EntityId, point_type> point_global_to_local(point_type const& x, int n_id = 0) {
    //        index_tuple i = (x - m_id_to_coordinates_shift_[n_id]) * 2;
    //
    //        EntityId s = pack(i) | m_id_to_shift_[n_id];
    //
    //        point_type r = (x - point(s)) / (_R * 2.0);
    //
    //        return std::make_tuple(s, r);
    //    }
    //
    //    static point_type point_local_to_global(EntityId s, point_type const& x) { return point(s) + x * _R * 2; }
    //
    //    static point_type point_local_to_global(std::tuple<EntityId, point_type> const& t) {
    //        return point_local_to_global(std::get<0>(t), std::get<1>(t));
    //    }

    //! @name id auxiliary functions
    //! @{
    static constexpr EntityId m_num_to_id_[] = {  //
        {.w = 1, .x = 0, .y = 0, .z = 0},
        {.w = 2, .x = 0, .y = 0, .z = 0},
        {.w = 4, .x = 0, .y = 0, .z = 0}};

    static EntityId DI(int n) { return m_num_to_id_[n]; }

    static EntityId DI(int n, EntityId s) { return EntityId{.v = s.v & m_num_to_id_[n].v}; }

    static EntityId dual(EntityId s) { return EntityId{.v = (s.v & (~_DA.v)) | ((~(s.v & _DA.v)) & _DA.v)}; }

    static EntityId delta_index(EntityId s) { return EntityId{.v = static_cast<int64_t>(s.v & _DA.v)}; }

    static EntityId rotate(EntityId s) {
        return EntityId{.w = static_cast<int16_t>(s.w),
                        .z = static_cast<int16_t>((s.z & ~0x1) | (s.y & 0x1)),
                        .y = static_cast<int16_t>((s.y & ~0x1) | (s.x & 0x1)),
                        .x = static_cast<int16_t>((s.x & ~0x1) | (s.z & 0x1))};
    }

    static EntityId inverse_rotate(EntityId s) {
        return EntityId{.w = static_cast<int16_t>(s.w),
                        .z = static_cast<int16_t>((s.z & ~0x1) | (s.x & 0x1)),
                        .y = static_cast<int16_t>((s.y & ~0x1) | (s.z & 0x1)),
                        .x = static_cast<int16_t>((s.x & ~0x1) | (s.y & 0x1))};
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
    enum node_id_tag {
        TAG_VERTEX = 0,
        TAG_EDGE0 = 1,
        TAG_EDGE1 = 2,
        TAG_EDGE2 = 4,
        TAG_FACE0 = 6,
        TAG_FACE1 = 5,
        TAG_FACE2 = 3,
        TAG_VOLUME = 7
    };

    static constexpr int node_id(EntityId s) {
        return s.w & 0x7;
        //                (s.x & 0x1) | ((s.y & 0x1) << 1) | ((s.z & 0x1) << 2);
    }

    static constexpr int m_id_to_index_[8] = {
        //

        0,  // 000
        0,  // 001
        1,  // 010
        2,  // 011
        2,  // 100
        1,  // 101
        0,  // 110
        0,  // 111
    };

    static constexpr int sub_index(EntityId s) { return m_id_to_index_[node_id(s)]; }

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

    static constexpr int m_adjacent_cell_num_[4 /* to GetIFORM*/][8 /* node id*/] =

        {  // VERTEX
            {/* 000*/ 1,
             /* 001*/ 2,
             /* 010*/ 2,
             /* 011*/ 4,
             /* 100*/ 2,
             /* 101*/ 4,
             /* 110*/ 4,
             /* 111*/ 8},

            // EDGE
            {/* 000*/ 6,
             /* 001*/ 1,
             /* 010*/ 1,
             /* 011*/ 4,
             /* 100*/ 1,
             /* 101*/ 4,
             /* 110*/ 4,
             /* 111*/ 12},

            // FACE
            {/* 000*/ 12,
             /* 001*/ 4,
             /* 010*/ 4,
             /* 011*/ 1,
             /* 100*/ 4,
             /* 101*/ 1,
             /* 110*/ 1,
             /* 111*/ 6},

            // VOLUME
            {/* 000*/ 8,
             /* 001*/ 4,
             /* 010*/ 4,
             /* 011*/ 2,
             /* 100*/ 4,
             /* 101*/ 2,
             /* 110*/ 2,
             /* 111*/ 1}

    };

    static constexpr EntityId m_adjacent_cell_matrix_[4 /* to GetIFORM*/][NUM_OF_NODE_ID /* node id*/]
                                                     [MAX_NUM_OF_ADJACENT_CELL /*id shift*/] = {

                                                         {// To VERTEX

                                                          {_DA},                  /* 000*/
                                                          {_DA - _DI, _DA + _DI}, /* 001*/
                                                          {_DA - _DJ, _DA + _DJ}, /* 010*/
                                                          {
                                                              /* 011*/
                                                              _DA - _DI - _DJ, /* 000*/
                                                              _DA + _DI - _DJ, /* 001*/
                                                              _DA - _DI + _DJ, /* 010*/
                                                              _DA + _DI + _DJ  /* 011 */
                                                          },
                                                          {_DA - _DK, _DA + _DK}, /* 100*/
                                                          {
                                                              /* 101*/
                                                              _DA - _DK - _DI, /*000*/
                                                              _DA - _DK + _DI, /*001*/
                                                              _DA + _DK - _DI, /*100*/
                                                              _DA + _DK + _DI  /*101*/
                                                          },
                                                          {
                                                              /* 110*/
                                                              _DA - _DJ - _DK, /*000*/
                                                              _DA + _DJ - _DK, /*010*/
                                                              _DA - _DJ + _DK, /*100*/
                                                              _DA + _DJ + _DK  /*110*/
                                                          },
                                                          {
                                                              /* 111*/
                                                              _DA - _DK - _DJ - _DI, /*000*/
                                                              _DA - _DK - _DJ + _DI, /*001*/
                                                              _DA - _DK + _DJ - _DI, /*010*/
                                                              _DA - _DK + _DJ + _DI, /*011*/
                                                              _DA + _DK - _DJ - _DI, /*100*/
                                                              _DA + _DK - _DJ + _DI, /*101*/
                                                              _DA + _DK + _DJ - _DI, /*110*/
                                                              _DA + _DK + _DJ + _DI  /*111*/
                                                          }},

                                                         {// To EDGE
                                                          {_DA + _DI, _DA - _DI, _DA + _DJ, _DA - _DJ, _DA + _DK,
                                                           _DA - _DK},  ///* 000*/

                                                          {_DA},  // /* 001*/

                                                          {_DA},  //  /* 010*/

                                                          {_DA - _DJ, _DA + _DI, _DA + _DJ,
                                                           _DA - _DI},  //      /* 011*/

                                                          {_DA},  //     /* 100*/

                                                          {_DA - _DI, _DA + _DK, _DA + _DI, _DA - _DK},  //  /* 101*/

                                                          {_DA - _DK, _DA + _DJ, _DA + _DK, _DA - _DJ},  //    /* 110*/

                                                          {
                                                              /* 111*/
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
                                                              _DA + _DK - _DI   //
                                                          }},

                                                         {{
                                                              // To FACE
                                                              /* 000*/
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
                                                              _DA + _DK - _DI   //
                                                          },
                                                          {
                                                              /* 001*/

                                                              _DA - _DJ,  //
                                                              _DA + _DK,  //
                                                              _DA + _DJ,  //
                                                              _DA - _DK   //
                                                          },
                                                          /* 010*/
                                                          {

                                                              _DA - _DK,  //
                                                              _DA + _DI,  //
                                                              _DA + _DK,  //
                                                              _DA - _DI   //
                                                          },

                                                          {_DA}, /* 011*/

                                                          {
                                                              /* 100*/
                                                              _DA - _DI,  //
                                                              _DA + _DJ,  //
                                                              _DA + _DI,  //
                                                              _DA - _DJ   //
                                                          },
                                                          {_DA}, /* 101*/

                                                          {_DA}, /* 110*/

                                                          {
                                                              /* 111*/
                                                              _DA - _DI,  //
                                                              _DA - _DJ,  //
                                                              _DA - _DK,  //
                                                              _DA + _DI,  //
                                                              _DA + _DJ,  //
                                                              _DA + _DK   //
                                                          }},

                                                         {{
                                                              // TO VOLUME   /* 000*/
                                                              _DA - _DI - _DJ - _DK,  //
                                                              _DA - _DI + _DJ - _DK,  //
                                                              _DA - _DI - _DJ + _DK,  //
                                                              _DA - _DI + _DJ + _DK,  //

                                                              _DA + _DI - _DJ - _DK,  //
                                                              _DA + _DI + _DJ - _DK,  //
                                                              _DA + _DI - _DJ + _DK,  //
                                                              _DA + _DI + _DJ + _DK   //

                                                          },

                                                          {
                                                              /* 001*/
                                                              _DA - _DJ - _DK,  //
                                                              _DA - _DJ + _DK,  //
                                                              _DA + _DJ - _DK,  //
                                                              _DA + _DJ + _DK   //
                                                          },

                                                          {
                                                              /* 010*/
                                                              _DA - _DK - _DI,  //
                                                              _DA - _DK + _DI,  //
                                                              _DA + _DK - _DI,  //
                                                              _DA + _DK + _DI   //
                                                          },

                                                          {_DA - _DK, _DA + _DK}, /* 011*/

                                                          {
                                                              /* 100*/
                                                              _DA - _DI - _DJ,  //
                                                              _DA - _DI + _DJ,  //
                                                              _DA + _DI - _DJ,  //
                                                              _DA + _DI + _DJ   //
                                                          },
                                                          {_DA - _DJ, _DA + _DJ}, /* 101*/
                                                          {_DA - _DI, _DA + _DI}, /* 110*/
                                                          {_DA}}                  /* 111*/

    };

    static int get_adjacent_entities(int IFORM, EntityId s, EntityId* res = nullptr) {
        return get_adjacent_entities(IFORM, node_id(s), s, res);
    }

    static int get_adjacent_entities(int IFORM, int nodeid, EntityId s, EntityId* res = nullptr) {
        if (res != nullptr) {
            for (int i = 0; i < m_adjacent_cell_num_[IFORM][nodeid]; ++i) {
                res[i] = s - _DA + m_adjacent_cell_matrix_[IFORM][nodeid][i];
            }
        }
        return m_adjacent_cell_num_[IFORM][nodeid];
    }
};

template <>
struct ContinueRange<EntityId> : public RangeBase<EntityId> {
   private:
    static constexpr int ndims = 3;

    SP_OBJECT_HEAD(ContinueRange<EntityId>, RangeBase<EntityId>)

   public:
    ContinueRange(index_type const* b = nullptr, index_type const* e = nullptr, int IFORM = VERTEX, int DOF = 1)
        : m_min_{b == nullptr ? 0 : b[0], b == nullptr ? 0 : b[1], b == nullptr ? 0 : b[2]},
          m_max_{e == nullptr ? 1 : e[0], e == nullptr ? 1 : e[1], e == nullptr ? 1 : e[2]},
          m_iform_(IFORM),
          m_dof_(DOF) {
        m_grain_size_ = 1;
        for (int i = 0; i < ndims; ++i) {
            if (m_max_[i] - m_min_[i] <= m_grain_size_[i]) { m_grain_size_[i] = m_max_[i] - m_min_[i]; }
        }
    }
    ContinueRange(index_tuple const& b, index_tuple const& e, int IFORM = VERTEX, int DOF = 1)
        : ContinueRange(&b[0], &(e[0]), IFORM, DOF) {}

    ContinueRange(std::tuple<index_tuple, index_tuple> const& b, int IFORM = VERTEX, int DOF = 1)
        : ContinueRange(std::get<0>(b), std::get<1>(b), IFORM, DOF) {}

    ContinueRange(this_type const& r)
        : m_min_(r.m_min_), m_max_(r.m_max_), m_grain_size_(r.m_grain_size_), m_iform_(r.m_iform_), m_dof_(r.m_dof_) {}

    std::shared_ptr<base_type> split(concept::tags::split const& proportion) override {
        auto res = std::make_shared<this_type>(*this);
        int n = 0;
        index_type L = m_max_[0] - m_min_[0];
        for (int i = 1; i < ndims; ++i) {
            if (m_max_[i] - m_min_[i] > L) {
                n = i;
                L = m_max_[i] - m_min_[i];
            }
        }

        m_max_[n] = m_min_[n] +
                    L * proportion.left() /
                        ((proportion.left() + proportion.right() > 0) ? (proportion.left() + proportion.right()) : 1);
        res->m_min_[n] = m_max_[n];

        return std::dynamic_pointer_cast<base_type>(res);
    }

    ~ContinueRange() {}

    void swap(this_type& other) {
        std::swap(m_iform_, other.m_iform_);
        std::swap(m_dof_, other.m_dof_);
        std::swap(m_min_, other.m_min_);
        std::swap(m_max_, other.m_max_);
        std::swap(m_grain_size_, other.m_grain_size_);
    }

    int entity_type() const { return m_iform_; }

    index_box_type index_box() const { return std::make_tuple(m_min_, m_max_); }

    bool empty() const override { return m_min_[0] == m_max_[0] || m_min_[1] == m_max_[1] || m_min_[2] == m_max_[2]; }

    size_t size() const override {
        return static_cast<size_t>(((m_iform_ == VERTEX || m_iform_ == VOLUME) ? 1 : 3) * (m_max_[0] - m_min_[0]) *
                                   (m_max_[1] - m_min_[1]) * (m_max_[2] - m_min_[2]));
    }

    // access
    index_tuple const& grainsize() const { return m_grain_size_; }

    bool is_divisible() const override {
        int count = 0;
        for (int i = 0; i < ndims; ++i) {
            if (m_max_[i] - m_min_[i] <= m_grain_size_[i]) { ++count; }
        }
        return count < ndims;
    }

    template <typename TFun>
    void foreach (TFun const& body) const {
        typedef EntityIdCoder M;
        ContinueRange const& r = *this;
        index_type ib = r.m_min_[0];
        index_type ie = r.m_max_[0];
        index_type ne = M::m_iform_to_num_of_ele_in_cell_[r.m_iform_];
#pragma omp parallel for
        for (index_type i = ib; i <= ie; ++i) {
            for (index_type j = r.m_min_[1], je = r.m_max_[1]; j <= je; ++j)
                for (index_type k = r.m_min_[2], ke = r.m_max_[2]; k <= ke; ++k)
                    for (index_type n = 0; n < ne; ++n) {
                        EntityId s;
                        s.x = static_cast<int16_t>(i);
                        s.y = static_cast<int16_t>(j);
                        s.z = static_cast<int16_t>(k);
                        s.w = static_cast<int16_t>(M::m_sub_index_to_id_[r.m_iform_][n]);
                        body(s);

                        // body(M::pack_index(i, j, k,M::m_sub_index_to_id_[r.m_iform_][n]));
                    }
        }
    }

    //    template <typename TFun>
    //    void foreach (
    //        TFun const& body,
    //        ENABLE_IF((simpla::concept::is_callable<TFun(index_type, index_type, index_type, index_type)>::value)))
    //        const {
    //        typedef EntityIdCoder M;
    //        ContinueRange const& r = *this;
    //        index_type ib = r.m_min_[0];
    //        index_type ie = r.m_max_[0];
    //        index_type ne = M::m_iform_to_num_of_ele_in_cell_[r.m_iform_];
    //#pragma omp parallel for
    //        for (index_type i = ib; i < ie; ++i) {
    //            for (index_type j = r.m_min_[1], je = r.m_max_[1]; j < je; ++j)
    //                for (index_type k = r.m_min_[2], ke = r.m_max_[2]; k < ke; ++k)
    //                    for (index_type n = 0; n < ne; ++n) { body(i, j, k, M::m_sub_index_to_id_[r.m_iform_][n]); }
    //        }
    //    }

   private:
    int m_iform_ = VERTEX;
    int m_dof_ = 1;

    index_tuple m_min_, m_max_, m_grain_size_;
};

template <>
struct UnorderedRange<EntityId> : public RangeBase<EntityId> {
    SP_OBJECT_HEAD(UnorderedRange<EntityId>, RangeBase<EntityId>)

   private:
    static constexpr int ndims = 3;
    int m_iform_ = VERTEX;
    std::set<EntityId> m_ids_;

   public:
    UnorderedRange(int iform = VERTEX) : m_iform_(iform) {}
    ~UnorderedRange() {}
    std::shared_ptr<base_type> split(concept::tags::split const& proportion) override {
        UNIMPLEMENTED;
        return (nullptr);
    }

    void swap(this_type& other) {
        std::swap(m_ids_, other.m_ids_);
        std::swap(m_iform_, other.m_iform_);
    }
    void Insert(EntityId s) { m_ids_.insert(s); }

    std::set<EntityId>& data() { return m_ids_; }
    std::set<EntityId> const& data() const { return m_ids_; }
    int entity_type() const { return m_iform_; }
    bool empty() const override { return m_ids_.empty(); }
    size_t size() const override { return m_ids_.size(); }
    bool is_divisible() const override { return false; }
    template <typename TFun>
    void foreach (TFun const& body, ENABLE_IF((simpla::concept::is_callable<TFun(EntityId)>::value))) const {
        for (auto s : m_ids_) { body(s); }
    }
};

}  // namespace simpla

#endif /* CORE_MESH_MESH_ENTITY_ID_CODER_H_ */
