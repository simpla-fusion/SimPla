//
// Created by salmon on 16-6-30.
//

#ifndef SIMPLA_MESHENTITYID_H
#define SIMPLA_MESHENTITYID_H

#include <cstdint>
#include "../sp_def.h"
#include "../gtl/nTuple.h"
#include "../parallel/Parallel.h"
#include "../parallel/ParallelTbb.h"

#include "MeshCommon.h"

namespace simpla { namespace mesh
{

/**
 *  @comment similar to MOAB::EntityHandle but using different code ruler and more efficienct for FD and SAMR  -- salmon. 2016.5.24
 *  @note different get_mesh should use different 'code and hash ruler'  -- salmon. 2016.5.24
 */


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
    * w =    |   |           |   |
    *        |   2-----------|---3
    *        |  /            |  /
    *        | /             | /
    *        |/              |/
    *        0---------------1   ---> x
    *
    *   \endverbatim
    */



constexpr inline bool operator==(MeshEntityId const &first, MeshEntityId const &second)
{
    return first.v == second.v;
}

constexpr inline MeshEntityId operator+(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{.v=first.v + second.v};
}

constexpr inline MeshEntityId operator|(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{.v=first.v | second.v};
}

constexpr inline MeshEntityId operator-(MeshEntityId const &first, MeshEntityId const &second)
{
    return MeshEntityId{
            static_cast<int16_t>(first.x - second.x),
            static_cast<int16_t>(first.y - second.y),
            static_cast<int16_t>(first.z - second.z),
            first.w};
}


template<int I = 1>
struct MeshEntityIdCoder_
{

    static constexpr int MAX_NUM_OF_NEIGHBOURS = 12;
    static constexpr int ndims = 3;

    typedef MeshEntityIdCoder_ this_type;

    typedef MeshEntityId id_type;


    static constexpr Real _R = 0.5;


    static constexpr id_type _DI{1, 0, 0, 0};
    static constexpr id_type _DJ{0, 1, 0, 0};
    static constexpr id_type _DK{0, 0, 1, 0};
    static constexpr id_type _DA{1, 1, 1, 0};


    static constexpr MeshEntityId delta_index(MeshEntityId s)
    {
        return MeshEntityId{.v=s.v & (~_DA.v)};
    };


    static constexpr MeshEntityId rotate(MeshEntityId s)
    {
        return MeshEntityId{
                static_cast<int16_t >((s.x & ~0x1) | (s.y & 0x1)),
                static_cast<int16_t >((s.y & ~0x1) | (s.z & 0x1)),
                static_cast<int16_t >((s.z & ~0x1) | (s.x & 0x1)),
                s.w
        };
    };

    static constexpr MeshEntityId inverse_rotate(MeshEntityId s)
    {
        return MeshEntityId{
                static_cast<int16_t >((s.x & ~0x1) | (s.z & 0x1)),
                static_cast<int16_t >((s.y & ~0x1) | (s.x & 0x1)),
                static_cast<int16_t >((s.z & ~0x1) | (s.y & 0x1)),
                s.w
        };
    };
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

    static constexpr int16_t m_id_to_shift_[] = {

            0,                    // 000
            1,                    // 001
            2,                    // 010
            3,                    // 011
            4,                    // 100
            5,                    // 101
            6,                    // 110
            7,                    // 111

    };

    static constexpr point_type m_id_to_coordinates_shift_[] = {

            {0,  0,  0},            // 000
            {_R, 0,  0},           // 001
            {0,  _R, 0},           // 010
            {0,  0,  _R},           // 011
            {_R, _R, 0},          // 100
            {_R, 0,  _R},          // 101
            {0,  _R, _R},          // 110
            {_R, _R, _R},         // 111

    };
    static constexpr int m_iform_to_num_of_ele_in_cell_[] = {
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
            3  // 111
    };


    template<int IFORM>
    static constexpr int sub_index_to_id(int n = 0)
    {
        return m_sub_index_to_id_[IFORM][n];
    }

    static constexpr int iform(id_type s)
    {
        return m_id_to_iform_[node_id(s)];
    }


    template<typename T>
    static constexpr id_type pack(T const &idx, int n_id = 0)
    {
        return id_type{static_cast<int16_t>(idx[0] << 1),

                       static_cast<int16_t>(idx[1] << 1),

                       static_cast<int16_t>(idx[2] << 1),

                       n_id};

        ;
    }

    static constexpr id_type pack_index(index_type i, index_type j, index_type k, index_type n_id = 0)
    {
        return id_type{static_cast<int16_t>(i << 1),
                       static_cast<int16_t>(j << 1),
                       static_cast<int16_t>(k << 1),
                       static_cast<int16_t>(n_id),
        };
    }

    static constexpr index_tuple unpack_index(id_type const &s)
    {
        return index_tuple{static_cast<index_type>(s.x),
                           static_cast<index_type>(s.y),
                           static_cast<index_type>(s.z)};
    }


    static point_type point(id_type const &s)
    {
        return point_type{static_cast<Real>(s.x) * _R,
                          static_cast<Real>(s.y) * _R,
                          static_cast<Real>(s.z) * _R,
        };
    }

    static std::tuple<MeshEntityId, point_type>
    point_global_to_local(point_type const &g, int nId = 0)
    {
        point_type r;
        r = g - m_id_to_coordinates_shift_[nId];
        MeshEntityId s{
                static_cast<int16_t >(r[0]),
                static_cast<int16_t >(r[1]),
                static_cast<int16_t >(r[2]),
                static_cast<int16_t >(nId)
        };

        r[0] -= static_cast<Real>(s.x);
        r[1] -= static_cast<Real>(s.y);
        r[2] -= static_cast<Real>(s.z);

        return std::make_tuple(s, r);
    }

    static constexpr int num_of_ele_in_cell(id_type s)
    {
        return m_id_to_num_of_ele_in_cell_[s.w & 0x7];
    }

//! @name id auxiliary functions
//! @{
    static constexpr id_type dual(id_type s)
    {
        return id_type{s.x, s.y, s.z, static_cast<int16_t >(((~s.w) & 0x7) | (s.w & (~0x7)))};
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

    static constexpr int16_t node_id(id_type const &s)
    {
        return static_cast<int16_t>(s.w & 0x7);
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

    static constexpr int sub_index(id_type const &s)
    {
        return m_id_to_index_[node_id(s)];
    }


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

    static constexpr id_type m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/] =
            {
                    //To VERTEX
                    {

                            /* 000*/
                            {//
                                    {1, 1, 1, 0}
                            },
                            /* 001*/
                            {       //
                                    {1, 0, 0, 0},
                                    {0, 0, 0, 0}
                            },
                            /* 010*/
                            {       //
                                    {0, 1, 0, 0},
                                    {0, 0, 0, 0}
                            },
                            /* 011*/
                            {//
                                    {0, 0, 0, 0},//_DA - _DI - _DJ, /* 000*/
                                    {0, 0, 1, 0},//_DA + _DI - _DJ, /* 001*/
                                    {0, 1, 0, 0},//_DA - _DI + _DJ, /* 010*/
                                    {0, 1, 1, 0} //_DA + _DI + _DJ /* 011 */
                            },
                            /* 100*/
                            {//
                                    {0, 0, 1, 0},
                                    {0, 0, 0, 0}
                            },
                            /* 101*/
                            {       //
                                    {0, 0, 0, 0},//_DA - _DK - _DI, /*000*/
                                    {0, 0, 1, 0},//_DA - _DK + _DI, /*001*/
                                    {1, 0, 0, 0},//_DA + _DK - _DI, /*100*/
                                    {1, 0, 1, 0} //_DA + _DK + _DI /*101*/
                            },
                            /* 110*/
                            {//
                                    {0, 0, 0, 0},//_DA - _DJ - _DK, /*000*/
                                    {0, 1, 0, 0},//_DA + _DJ - _DK, /*010*/
                                    {1, 0, 0, 0},//_DA - _DJ + _DK, /*100*/
                                    {1, 1, 0, 0} //_DA + _DJ + _DK /*110*/
                            },
                            /* 111*/
                            {       //
                                    {0, 0, 0, 0},//_DA - _DK - _DJ - _DI, /*000*/
                                    {0, 0, 1, 0},//_DA - _DK - _DJ + _DI, /*001*/
                                    {0, 1, 0, 0},//_DA - _DK + _DJ - _DI, /*010*/
                                    {0, 1, 1, 0},//_DA - _DK + _DJ + _DI, /*011*/
                                    {1, 0, 0, 0},//_DA + _DK - _DJ - _DI, /*100*/
                                    {1, 0, 1, 0},//_DA + _DK - _DJ + _DI, /*101*/
                                    {1, 1, 0, 0},//_DA + _DK + _DJ - _DI, /*110*/
                                    {1, 1, 1, 0} //_DA + _DK + _DJ + _DI  /*111*/

                            }

                    },

                    //To EDGE
                    {
                            /* 000*/
                            {       //
                                    {0, 0, 0, 1},   //_DA + _DI,
                                    {0, 0, -1, 1},  //_DA - _DI,
                                    {0, 1, 0, 2},   //_DA + _DJ,
                                    {0, -1, 0, 2},   //_DA - _DJ,
                                    {1, 0, 0, 4},   //_DA + _DK,
                                    {-1, 0, 0, 4},   //_DA - _DK
                            },
                            /* 001*/
                            {
                                    {0, 0, 0, 1}    //_DA
                            },
                            /* 010*/
                            {
                                    {0, 0, 0, 2}   // _DA
                            },
                            /* 011*/
                            {        //
                                    {0, 0, 0, 1},//_DA - _DJ,
                                    {0, 0, 1, 2},//_DA + _DI,
                                    {0, 1, 0, 1},//_DA + _DJ,
                                    {0, 0, 0, 2},//_DA - _DI
                            },
                            /* 100*/
                            {       //
                                    {0, 0, 0, 4}
                            },
                            /* 101*/
                            {         //
                                    {0, 0, 0, 1},//_DA - _DI,
                                    {0, 0, 1, 2},//_DA + _DK,
                                    {0, 1, 0, 1},//_DA + _DI,
                                    {0, 0, 0, 2},//_DA - _DK
                            },
//                            /* 110*/
//                            {       //
//                                    _DA - _DK,
//                                    _DA + _DJ,
//                                    _DA + _DK,
//                                    _DA - _DJ
//                            },
//                            /* 111*/
//                            {       //
//                                    _DA - _DK - _DJ,  //-> 001
//                                    _DA - _DK + _DI,  //   012
//                                    _DA - _DK + _DJ,  //   021
//                                    _DA - _DK - _DI,  //   010
//
//                                    _DA - _DI - _DJ,  //
//                                    _DA - _DI + _DJ,  //
//                                    _DA + _DI - _DJ,  //
//                                    _DA + _DI + _DJ,  //
//
//                                    _DA + _DK - _DJ,  //
//                                    _DA + _DK + _DI,  //
//                                    _DA + _DK + _DJ,  //
//                                    _DA + _DK - _DI  //
//                            }
                    },

//                    //To FACE
//                    {
//                            /* 000*/
//                            {       //
//                                    _DA - _DK - _DJ,  //
//                                    _DA - _DK + _DI,  //
//                                    _DA - _DK + _DJ,  //
//                                    _DA - _DK - _DI,  //
//
//                                    _DA - _DI - _DJ,  //
//                                    _DA - _DI + _DJ,  //
//                                    _DA + _DI - _DJ,  //
//                                    _DA + _DI + _DJ,  //
//
//                                    _DA + _DK - _DJ,  //
//                                    _DA + _DK + _DI,  //
//                                    _DA + _DK + _DJ,  //
//                                    _DA + _DK - _DI  //
//                            },
//                            /* 001*/
//                            {       //
//                                    _DA - _DJ,          //
//                                    _DA + _DK,   //
//                                    _DA + _DJ,   //
//                                    _DA - _DK    //
//                            },
//                            /* 010*/
//                            {       //
//                                    _DA - _DK,          //
//                                    _DA + _DI,   //
//                                    _DA + _DK,   //
//                                    _DA - _DI    //
//                            },
//                            /* 011*/
//                            {       _DA},
//                            /* 100*/
//                            {//
//                                    _DA - _DI,         //
//                                    _DA + _DJ,  //
//                                    _DA + _DI,  //
//                                    _DA - _DJ   //
//                            },
//                            /* 101*/
//                            {       //
//                                    _DA
//                            },
//                            /* 110*/
//                            {       //
//                                    _DA
//                            },
//                            /* 111*/
//                            {       //
//                                    _DA - _DI,         //
//                                    _DA - _DJ,  //
//                                    _DA - _DK,  //
//                                    _DA + _DI,  //
//                                    _DA + _DJ,  //
//                                    _DA + _DK   //
//                            }},
//                    // TO VOLUME
//                    {
//                            /* 000*/
//                            {       //
//                                    _DA - _DI - _DJ - _DK,  //
//                                    _DA - _DI + _DJ - _DK,  //
//                                    _DA - _DI - _DJ + _DK,  //
//                                    _DA - _DI + _DJ + _DK,  //
//
//                                    _DA + _DI - _DJ - _DK,  //
//                                    _DA + _DI + _DJ - _DK,  //
//                                    _DA + _DI - _DJ + _DK,  //
//                                    _DA + _DI + _DJ + _DK  //
//
//                            },
//                            /* 001*/
//                            {       //
//                                    _DA - _DJ - _DK,           //
//                                    _DA - _DJ + _DK,    //
//                                    _DA + _DJ - _DK,    //
//                                    _DA + _DJ + _DK     //
//                            },
//                            /* 010*/
//                            {        //
//                                    _DA - _DK - _DI,  //
//                                    _DA - _DK + _DI,  //
//                                    _DA + _DK - _DI,  //
//                                    _DA + _DK + _DI   //
//                            },
//                            /* 011*/
//                            {       //
//                                    _DA - _DK,
//                                    _DA + _DK},
//                            /* 100*/
//                            {         //
//                                    _DA - _DI - _DJ,   //
//                                    _DA - _DI + _DJ,   //
//                                    _DA + _DI - _DJ,   //
//                                    _DA + _DI + _DJ    //
//                            },
//                            /* 101*/
//                            {//
//                                    _DA - _DJ,
//                                    _DA + _DJ
//                            },
//                            /* 110*/
//                            {       //
//                                    _DA - _DI,
//                                    _DA + _DI
//                            },
//                            /* 111*/
//                            {//
//                                    _DA
//                            }
//                    }

            };

    static int get_adjacent_entities(int IFORM, id_type s, id_type *res = nullptr)
    {
        return get_adjacent_entities(IFORM, node_id(s), s, res);
    }

    static int get_adjacent_entities(int IFORM, int nodeid, id_type s, id_type *res = nullptr)
    {
        if (res != nullptr)
        {
            for (int i = 0; i < m_adjacent_cell_num_[IFORM][nodeid]; ++i)
            {
                res[i] = m_adjacent_cell_matrix_[IFORM][nodeid][i] + s;
            }
        }
        return m_adjacent_cell_num_[IFORM][nodeid];
    }

    struct range_type
    {
    private:
        typedef range_type this_type;
    public:

//        typedef iterator const_iterator;

        range_type() : m_iform_(VERTEX), m_min_(), m_max_(m_min_), m_grain_size_(m_min_) { }

        // constructors

        range_type(index_tuple const &b, index_tuple const &e, int IFORM = 0)
                : m_iform_(IFORM), m_min_(b), m_max_(e)
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
                : m_iform_(r.m_iform_), m_min_(r.m_min_), m_max_(r.m_max_), m_grain_size_(r.m_grain_size_)
        {
        }

        template<typename T0, typename T1, typename T2>
        range_type(T0 const &b, T1 const &e, index_tuple const &grain_size, int IFORM = 0)
                : m_iform_(IFORM), m_min_(b), m_max_(e), m_grain_size_(grain_size)
        {
        }


        range_type(range_type &r, parallel::tags::split)
                : m_iform_(r.m_iform_), m_min_(r.m_min_), m_max_(r.m_max_), m_grain_size_(r.m_grain_size_)
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


        range_type(this_type &r, parallel::tags::proportional_split const &proportion)
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


        ~range_type() { }


        void swap(this_type &other)
        {
            std::swap(m_iform_, other.m_iform_);
            std::swap(m_min_, other.m_min_);
            std::swap(m_max_, other.m_max_);
            std::swap(m_grain_size_, other.m_grain_size_);
        }

        int entity_type() const { return static_cast<int>(m_iform_); }

        std::tuple<index_tuple, index_tuple> index_box() const { return std::make_tuple(m_min_, m_max_); }

        // Proportional split is enabled
        static const bool is_splittable_in_proportion = true;

        // capacity


        bool empty() const { return m_min_ == m_max_; }

        size_t size() const
        {
            return ((m_iform_ == 0 || m_iform_ == 3) ? 1 : 3) * NProduct(m_max_ - m_min_);
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

        // iterators
//        const_iterator begin() const { return const_iterator(m_min_, m_min_, m_max_, m_iform_); }
//        const_iterator end() const { return const_iterator(m_min_, m_min_, m_max_, m_iform_).end(); }


        template<typename Body>
        void parallel_foreach(Body const &body) const
        {
#ifdef USE_TBB
            tbb::parallel_for(*this, [&](range_type const &r)
            {
#else
                range_type const &r = *this;
#   ifdef  _OPENMP
#           pragma omp parallel for
#   endif
#endif
                for (index_type i = r.m_min_[0], ie = r.m_max_[0]; i < ie; ++i)
                    for (index_type j = r.m_min_[1], je = r.m_max_[1]; j < je; ++j)
                        for (index_type k = r.m_min_[2], ke = r.m_max_[2]; k < ke; ++k)
                            for (index_type n = 0, ne = m_iform_to_num_of_ele_in_cell_[r.m_iform_]; n < ne; ++n)
                            {
                                body(pack_index(i, j, k, m_sub_index_to_id_[r.m_iform_][n]));
                            }
#ifdef USE_TBB
            });
#endif
        }

        template<typename Body>
        void serial_foreach(Body const &body) const
        {
            range_type const &r = *this;
            for (index_type i = r.m_min_[0], ie = r.m_max_[0]; i < ie; ++i)
                for (index_type j = r.m_min_[1], je = r.m_max_[1]; j < je; ++j)
                    for (index_type k = r.m_min_[2], ke = r.m_max_[2]; k < ke; ++k)
                        for (index_type n = 0, ne = m_iform_to_num_of_ele_in_cell_[r.m_iform_]; n < ne; ++n)
                        {
                            body(pack_index(i, j, k, m_sub_index_to_id_[r.m_iform_][n]));
                        }

        }

        template<typename Body>
        void foreach(Body const &body, bool auto_parallel = false) const
        {
            if (auto_parallel)
            {
                parallel_foreach(body);
            }
            else
            {
                serial_foreach(body);
            }
        }

    private:


        int m_iform_;
        index_tuple m_min_, m_max_, m_grain_size_;
    };

//    typedef RangeHolder<iterator> range_type;

    template<typename T0, typename T1>
    static range_type make_range(T0 const &min, T1 const &max, int iform = 0)
    {
        return range_type(min, max, iform);
    }

    template<int IFORM, typename TB>
    static range_type make_range(TB const &b)
    {
        return make_range(traits::get<0>(b), traits::get<1>(b), IFORM);
    }


    template<int IFORM, typename T0, typename T1>
    static range_type make_range(T0 const &b, T1 const &e)
    {
        return make_range(b, e, IFORM);
    }


    static index_type hash(id_type const &s, index_tuple const &b, index_tuple const &e)
    {
        //C-ORDER SLOW FIRST

        return
                (
                        (s.z + e[2] - b[2] - b[2]) % (e[2] - b[2]) +

                        (
                                ((s.y + e[1] - b[1] - b[1]) % (e[1] - b[1])) +

                                ((s.x + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])

                        ) * (e[2] - b[2])

                ) * num_of_ele_in_cell(s) + sub_index(s);

    }


    static size_type max_hash(index_tuple const &b, index_tuple const &e, int IFORM)
    {
        return static_cast<size_type>(((e[0] - b[0]) * (e[1] - b[1]) * (e[2] - b[2])) *
                                      m_id_to_num_of_ele_in_cell_[m_sub_index_to_id_[IFORM][0]]);
    }

//    template<typename TGeometry>
//    static void get_element_volume_in_cell(TGeometry const &geo, id_type s0, Real *v, Real *inv_v, Real *dual_v,
//                                           Real *inv_dual_v)
//    {
//
//        /**
//         *\verbatim
//         *                ^y
//         *               /
//         *        z     /
//         *        ^    /
//         *        |   6---------------7
//         *        |  /|              /|
//         *        | / |             / |
//         *        |/  |            /  |
//         *        4---|-----------5   |
//         *        |   |           |   |
//         *        |   2-----------|---3
//         *        |  /            |  /
//         *        | /             | /
//         *        |/              |/
//         *        0---------------1---> x
//         *
//         *\endverbatim
//         */
//
//
//
//        typedef typename TGeometry::point_type point_type;
//
//        auto dims = geo.dimensions();
//
//
//        static constexpr id_type HI = 1UL << (MESH_RESOLUTION);
//        static constexpr id_type HJ = HI << ID_DIGITS;
//        static constexpr id_type HK = HI << (ID_DIGITS * 2);
//
//        static constexpr id_type HA = HI | HJ | HK;
//        //primary
//        {
//            size_t s = (s0 | FULL_OVERFLOW_FLAG);
//
//            point_type p[NUM_OF_NODE_ID] = {
//
//                    /*000*/  geo.point(s),                       //
//                    /*001*/  geo.point(s + (HI)),          //
//                    /*010*/  geo.point(s + (HJ)),          //
//                    /*011*/  geo.point(s + (HJ | HI)),  //
//
//                    /*100*/  geo.point(s + (HK)),          //
//                    /*101*/  geo.point(s + (HK | HI)),   //
//                    /*110*/  geo.point(s + (HK | HJ)),   //
//                    /*111*/  geo.point(s + (HK | HJ | HI))    //
//
//            };
//
//
//            v[TAG_VERTEX] = 1;
//
//            v[TAG_EDGE0] = geo.simplex_length(p[0], p[1]);
//            v[TAG_EDGE1] = geo.simplex_length(p[0], p[2]);
//            v[TAG_EDGE2] = geo.simplex_length(p[0], p[4]);
//
//            v[TAG_FACE0] = geo.simplex_area(p[0], p[2], p[6]) + geo.simplex_area(p[0], p[6], p[4]);
//            v[TAG_FACE1] = geo.simplex_area(p[0], p[1], p[5]) + geo.simplex_area(p[0], p[5], p[4]);
//            v[TAG_FACE2] = geo.simplex_area(p[0], p[1], p[3]) + geo.simplex_area(p[0], p[3], p[2]);
//
//
//            v[TAG_VOLUME] = geo.simplex_volume(p[0], p[1], p[2], p[4]) + //
//                            geo.simplex_volume(p[1], p[4], p[5], p[2]) + //
//                            geo.simplex_volume(p[2], p[6], p[4], p[5]) + //
//                            geo.simplex_volume(p[1], p[3], p[2], p[5]) + //
//                            geo.simplex_volume(p[3], p[5], p[7], p[6]) + //
//                            geo.simplex_volume(p[3], p[6], p[2], p[5]);
//
//        }
//        //dual
//        {
//            size_t s = (s0 | FULL_OVERFLOW_FLAG) - (HA >> 1);
//
////            point_type p[NUM_OF_NODE_ID] = {
////
////                    /*000*/    geo.point(s + ((LK | LJ | LI) << 1)),   //
////                    /*001*/    geo.point(s + ((LK | LJ | HI) << 1)),   //
////                    /*010*/    geo.point(s + ((LK | HJ | LI) << 1)),   //
////                    /*011*/    geo.point(s + ((LK | HJ | HI) << 1)),   //
////
////                    /*100*/    geo.point(s + ((HK | LJ | LI) << 1)),   //
////                    /*101*/    geo.point(s + ((HK | LJ | HI) << 1)),   //
////                    /*110*/    geo.point(s + ((HK | HJ | LI) << 1)),   //
////                    /*111*/    geo.point(s + ((HK | HJ | HI) << 1))    //
////
////            };
//
//            point_type p[NUM_OF_NODE_ID] = {
//
//                    /*000*/  geo.point(s),                       //
//                    /*001*/  geo.point(s + (HI)),          //
//                    /*010*/  geo.point(s + (HJ)),          //
//                    /*011*/  geo.point(s + (HJ | HI)),  //
//
//                    /*100*/  geo.point(s + (HK)),          //
//                    /*101*/  geo.point(s + (HK | HI)),   //
//                    /*110*/  geo.point(s + (HK | HJ)),   //
//                    /*111*/  geo.point(s + (HK | HJ | HI))    //
//
//            };
//
//
//            dual_v[TAG_VOLUME] = 1;
//
//            dual_v[TAG_FACE0] = geo.simplex_length(p[6], p[7]);
//            dual_v[TAG_FACE1] = geo.simplex_length(p[5], p[7]);
//            dual_v[TAG_FACE2] = geo.simplex_length(p[3], p[7]);
//
//
//            dual_v[TAG_EDGE0] = geo.simplex_area(p[1], p[3], p[5]) + geo.simplex_area(p[3], p[7], p[5]);
//            dual_v[TAG_EDGE1] = geo.simplex_area(p[2], p[3], p[7]) + geo.simplex_area(p[2], p[7], p[6]);
//            dual_v[TAG_EDGE2] = geo.simplex_area(p[4], p[5], p[7]) + geo.simplex_area(p[4], p[7], p[6]);
//
//
//            dual_v[TAG_VERTEX] = geo.simplex_volume(p[0], p[1], p[2], p[4]) + //
//                                 geo.simplex_volume(p[1], p[4], p[5], p[2]) + //
//                                 geo.simplex_volume(p[2], p[6], p[4], p[5]) + //
//                                 geo.simplex_volume(p[1], p[3], p[2], p[5]) +
//                                 geo.simplex_volume(p[3], p[5], p[7], p[6]) + //
//                                 geo.simplex_volume(p[3], p[6], p[2], p[5])  //
//                    ;
//
//        }
//
//        for (int i = 0; i < NUM_OF_NODE_ID; ++i)
//        {
//            inv_v[i] = 1.0 / v[i];
//            inv_dual_v[i] = 1.0 / dual_v[i];
//        }
//
//
//        if (dims[0] <= 1)
//        {
//            inv_v[TAG_EDGE0] = 0;
//            inv_dual_v[TAG_FACE0] = 0;
//        }
//
//        if (dims[1] <= 1)
//        {
//            inv_v[TAG_EDGE1] = 0;
//            inv_dual_v[TAG_FACE1] = 0;
//        }
//
//        if (dims[2] <= 1)
//        {
//            inv_v[TAG_EDGE2] = 0;
//
//            inv_dual_v[TAG_FACE2] = 0;
//        }
//
//
//    }
};


/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<int L> constexpr int MeshEntityIdCoder_<L>::ndims;

template<int L> constexpr Real MeshEntityIdCoder_<L>::_R;
template<int L> constexpr typename MeshEntityIdCoder_<L>::id_type MeshEntityIdCoder_<L>::_DK;
template<int L> constexpr typename MeshEntityIdCoder_<L>::id_type MeshEntityIdCoder_<L>::_DJ;
template<int L> constexpr typename MeshEntityIdCoder_<L>::id_type MeshEntityIdCoder_<L>::_DI;
template<int L> constexpr typename MeshEntityIdCoder_<L>::id_type MeshEntityIdCoder_<L>::_DA;
template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_index_[];
template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_iform_[];
template<int L> constexpr int MeshEntityIdCoder_<L>::m_id_to_num_of_ele_in_cell_[];
template<int L> constexpr int MeshEntityIdCoder_<L>::m_adjacent_cell_num_[4][8];
template<int L> constexpr int MeshEntityIdCoder_<L>::m_iform_to_num_of_ele_in_cell_[];
template<int L> constexpr int16_t MeshEntityIdCoder_<L>::m_id_to_shift_[];
template<int L> constexpr int MeshEntityIdCoder_<L>::m_sub_index_to_id_[4][3];
template<int L> constexpr MeshEntityId MeshEntityIdCoder_<L>::m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/];
template<int L> constexpr point_type MeshEntityIdCoder_<L>::m_id_to_coordinates_shift_[];

typedef MeshEntityIdCoder_<1> MeshEntityIdCoder;
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHENTITYID_H
