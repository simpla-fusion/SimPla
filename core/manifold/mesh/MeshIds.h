/**
 * @file mesh_ids.h
 *
 * @date 2015-3-19
 * @author salmon
 */

#ifndef CORE_MESH_MESH_IDS_H_
#define CORE_MESH_MESH_IDS_H_

#include <stddef.h>
#include <limits>
#include <tuple>

#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"
#include "../../gtl/iterator/block_iterator.h"
#include "../../parallel/parallel.h"
#include "../manifold_traits.h"

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
 */
struct MeshIDs
{
    /// @name level independent
    /// @{

    static constexpr int MAX_NUM_OF_NEIGHBOURS = 12;
    static constexpr int ndims = 3;
    static constexpr int MESH_RESOLUTION = 1;

    typedef MeshIDs this_type;

    typedef std::uint64_t id_type;

    typedef nTuple<id_type, ndims> id_tuple;

    typedef nTuple<Real, ndims> coordinates_tuple;


    typedef long index_type;

    typedef long difference_type;

    typedef nTuple<index_type, ndims> index_tuple;

//	typedef Real coordinate_type;

    static constexpr int FULL_DIGITS = std::numeric_limits<id_type>::digits;

    static constexpr int ID_DIGITS = 21;

    static constexpr int HEAD_DIGITS = (FULL_DIGITS - ID_DIGITS * 3);

    static constexpr id_type ID_MASK = (1UL << ID_DIGITS) - 1;

    static constexpr id_type NO_HAED = (1UL << (ID_DIGITS * 3)) - 1;

    static constexpr id_type OVERFLOW_FLAG = (1UL) << (ID_DIGITS - 1);

    static constexpr id_type FULL_OVERFLOW_FLAG = OVERFLOW_FLAG
                                                  | (OVERFLOW_FLAG << ID_DIGITS) |
                                                  (OVERFLOW_FLAG << (ID_DIGITS * 2));

    static constexpr id_type INDEX_ZERO = (1UL) << (ID_DIGITS - 2);

    static constexpr id_type ID_ZERO = INDEX_ZERO | (INDEX_ZERO << ID_DIGITS)
                                       | (INDEX_ZERO << (ID_DIGITS * 2));

    static constexpr Real EPSILON = 1.0 / static_cast<Real>(INDEX_ZERO);

    /// @}

    /// @name level dependent
    /// @{

    static constexpr id_type SUB_ID_MASK = ((1UL << MESH_RESOLUTION) - 1);

    static constexpr id_type _D = 1UL << (MESH_RESOLUTION - 1);

    static constexpr Real _R = static_cast<Real>(_D);

    static constexpr id_type _DI = _D;

    static constexpr id_type _DJ = _D << (ID_DIGITS);

    static constexpr id_type _DK = _D << (ID_DIGITS * 2);

    static constexpr id_type _DA = _DI | _DJ | _DK;


    static constexpr id_type PRIMARY_ID_MASK_ = ID_MASK & (~SUB_ID_MASK);

    static constexpr id_type PRIMARY_ID_MASK = PRIMARY_ID_MASK_
                                               | (PRIMARY_ID_MASK_ << ID_DIGITS)
                                               | (PRIMARY_ID_MASK_ << (ID_DIGITS * 2));


    static constexpr Real COORDINATES_MESH_FACTOR = static_cast<Real>(1UL << MESH_RESOLUTION);
    static constexpr Real MESH_COORDINATES_FACTOR = 1.0 / COORDINATES_MESH_FACTOR;

    /// @}
    static constexpr Vec3 dx()
    {
        return Vec3({COORDINATES_MESH_FACTOR, COORDINATES_MESH_FACTOR, COORDINATES_MESH_FACTOR});
    }

    static constexpr int m_sub_index_to_id_[4][3] = { //

            {0, 0, 0}, /*VERTEX*/
            {1, 2, 4}, /*EDGE*/
            {6, 5, 3}, /*FACE*/
            {7, 7, 7} /*VOLUME*/

    };

    static constexpr id_type m_id_to_sub_index_[8] = { //

            0, // 000
            0, // 001
            1, // 010
            2, // 011
            2, // 100
            1, // 101
            0, // 110
            0, // 111
    };

    static constexpr id_type m_id_to_shift_[] = {

            0,                    // 000
            _DI,                    // 001
            _DJ,                    // 010
            (_DI | _DJ),                    // 011
            _DK,                    // 100
            (_DK | _DI),                    // 101
            (_DJ | _DK),                    // 110
            _DA                    // 111

    };

    static constexpr coordinates_tuple m_id_to_coordinates_shift_[] = {

            {0, 0, 0},            // 000
            {_R, 0, 0},           // 001
            {0, _R, 0},           // 010
            {0, 0, _R},           // 011
            {_R, _R, 0},          // 100
            {_R, 0, _R},          // 101
            {0, _R, _R},          // 110
            {0, _R, _R},          // 111

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

            VERTEX, // 000
            EDGE, // 001
            EDGE, // 010
            FACE, // 011
            EDGE, // 100
            FACE, // 101
            FACE, // 110
            VOLUME // 111
    };

    template<size_t IFORM>
    static constexpr int sub_index_to_id(int n = 0)
    {
        return m_sub_index_to_id_[IFORM][n];
    }

    static constexpr int iform(id_type s)
    {
        return m_id_to_iform_[node_id(s)];
    }

    static constexpr id_type pack(id_type i0, id_type i1, id_type i2)
    {
        return (i0 & ID_MASK) | ((i1 & ID_MASK) << ID_DIGITS) | ((i2 & ID_MASK) << (ID_DIGITS * 2)) |
               FULL_OVERFLOW_FLAG;
    }

    template<typename T>
    static constexpr id_type pack(T const &idx)
    {
        return pack(static_cast<id_type>(idx[0]),

                    static_cast<id_type>(idx[1]),

                    static_cast<id_type>(idx[2]));
    }


    template<typename T>
    static constexpr id_type pack_index(T const &idx, int n_id = 0)
    {

        return pack(static_cast<id_type>(idx[0]) << MESH_RESOLUTION,

                    static_cast<id_type>(idx[1]) << MESH_RESOLUTION,

                    static_cast<id_type>(idx[2]) << MESH_RESOLUTION) | m_id_to_shift_[n_id];
    }

    static constexpr id_type extent_flag_bit(id_type const &s, int n = ID_DIGITS - 2)
    {
        return s | (((s & (1UL << n)) == 0) ? 0UL : (static_cast<id_type>( -1L << (n + 1))));
    }

    static constexpr id_type unpack_id(id_type const &s, int n)
    {
        return extent_flag_bit(((s & (~FULL_OVERFLOW_FLAG)) >> (ID_DIGITS * n)) & ID_MASK);
    }

    static constexpr index_type unpack_index(id_type const &s, int n)
    {
        return static_cast<index_type>(extent_flag_bit(
                (((s & (~FULL_OVERFLOW_FLAG)) >> (ID_DIGITS * n)) & ID_MASK) >> MESH_RESOLUTION,
                ID_DIGITS - 2 - MESH_RESOLUTION));
    }


    static constexpr id_tuple unpack(id_type s)
    {
        return id_tuple({unpack_id(s, 0), unpack_id(s, 1), unpack_id(s, 2)});;
    }

    static constexpr index_tuple unpack_index(id_type s)
    {
        return index_tuple({unpack_index(s, 0), unpack_index(s, 1), unpack_index(s, 2)});
    }


    template<typename T>
    static constexpr T type_cast(id_type s)
    {
        return static_cast<T>(unpack(s));
    }

    static coordinates_tuple coordinates(id_type const &s)
    {
        return coordinates_tuple{static_cast<Real>(static_cast<index_type>(unpack_id(s, 0))),
                static_cast<Real>(static_cast<index_type>(unpack_id(s, 1))),
                static_cast<Real>(static_cast<index_type>(unpack_id(s, 2)))
        };
    }

    static coordinates_tuple point(id_type const &s)
    {
        return coordinates_tuple{static_cast<Real>(static_cast<index_type>(unpack_id(s, 0))),
                static_cast<Real>(static_cast<index_type>(unpack_id(s, 1))),
                static_cast<Real>(static_cast<index_type>(unpack_id(s, 2)))
        };
    }


    static coordinates_tuple point(nTuple<index_type, ndims> const &idx)
    {
        return coordinates_tuple{
                static_cast<Real>((idx[0] << MESH_RESOLUTION)),
                static_cast<Real>((idx[1] << MESH_RESOLUTION)),
                static_cast<Real>((idx[2] << MESH_RESOLUTION))
        };
    }


    static constexpr int num_of_ele_in_cell(id_type s)
    {
        return m_id_to_num_of_ele_in_cell_[node_id(s)];
    }

    template<typename TX>
    static std::tuple<id_type, coordinates_tuple> coordinates_global_to_local(
            TX const &x, int n_id = 0)
    {

        id_type s = (pack(x - m_id_to_coordinates_shift_[n_id])
                     & PRIMARY_ID_MASK) | m_id_to_shift_[n_id];

        coordinates_tuple r;

        r = (x - coordinates(s)) / (_R * 2.0);

        return std::make_tuple(s, r);

    }

    static coordinates_tuple coordinates_local_to_global(
            std::tuple<id_type, coordinates_tuple> const &t)
    {
        return point(std::get<0>(t)) + std::get<1>(t);
    }

//! @name id auxiliary functions
//! @{
    static constexpr id_type dual(id_type s)
    {
        return (s & (~_DA)) | ((~(s & _DA)) & _DA);

    }

    static constexpr id_type DI(int n, id_type s)
    {
        return (s >> (n * ID_DIGITS)) & _D;
    }


    static constexpr id_type delta_index(id_type s)
    {
        return (s & _DA);
    }

    static constexpr id_type rotate(id_type const &s)
    {
        return ((s & (~_DA))
                | (((s & (_DA)) << ID_DIGITS) | ((s & _DK) >> (ID_DIGITS * 2))))
               & NO_HAED;
    }

    static constexpr id_type inverse_rotate(id_type const &s)
    {
        return ((s & (~_DA))
                | (((s & (_DA)) >> ID_DIGITS) | ((s & _DI) << (ID_DIGITS * 2))))
               & NO_HAED;
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
     *        |  / EDGE1        |  /
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

    static constexpr size_t node_id(id_type const &s)
    {
        return ((s >> (MESH_RESOLUTION - 1)) & 1UL)
               | ((s >> (ID_DIGITS + MESH_RESOLUTION - 2)) & 2UL)
               | ((s >> (ID_DIGITS * 2 + MESH_RESOLUTION - 3)) & 4UL);
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

    static constexpr id_type m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/] =
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

    static int get_adjacent_cells(size_t IFORM, id_type s, id_type *res = nullptr)
    {
        return get_adjacent_cells(IFORM, node_id(s), s, res);
    }

    static int get_adjacent_cells(size_t IFORM, size_t nodeid, id_type s, id_type *res = nullptr)
    {
        if (res != nullptr)
        {
            for (int i = 0; i < m_adjacent_cell_num_[IFORM][nodeid]; ++i)
            {
                res[i] = (((s | FULL_OVERFLOW_FLAG) - _DA + m_adjacent_cell_matrix_[IFORM][nodeid][i])) |
                         (FULL_OVERFLOW_FLAG);
            }
        }
        return m_adjacent_cell_num_[IFORM][nodeid];
    }

    struct iterator : public block_iterator<index_type, MeshIDs::ndims + 1>
    {
    private:
        static constexpr int ndims = MeshIDs::ndims;
        typedef block_iterator<index_type, ndims + 1> base_type;

        int m_iform_;

    public:
        iterator() : base_type(), m_iform_(VERTEX) { }

        iterator(id_type s, id_type b, id_type e)
                : base_type(unpack_index(s), unpack_index(b), unpack_index(e)), m_iform_(iform(s))
        {
            nTuple<index_type, ndims + 1> self, min, max;
            self = unpack_index(s);
            min = unpack_index(b);
            max = unpack_index(e);

            self[ndims] = 0;

            min[ndims] = 0;

            max[ndims] = num_of_ele_in_cell(s);

            base_type(self, min, max).swap(*this);

        }

        template<typename T0, typename T1, typename T2>
        iterator(T0 const &pself, T1 const &pmin, T2 const &pmax, int IFORM = VERTEX) :
                m_iform_(IFORM)
        {
            nTuple<index_type, ndims + 1> self, min, max;
            self = pself;
            min = pmin;
            max = pmax;

            self[ndims] = 0;

            min[ndims] = 0;

            max[ndims] = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;

            base_type(self, min, max).swap(*this);
        }

        iterator(iterator const &other) : base_type(other), m_iform_(other.m_iform_)
        {
        }

        iterator(iterator &&other) : base_type(other), m_iform_(other.m_iform_)
        {
        }

        iterator(base_type const &other, int IFORM) : base_type(other), m_iform_(IFORM)
        {
        }

        ~iterator() { }

        iterator end() const
        {
            return iterator(base_type::end(), m_iform_);
        }

        iterator &operator=(iterator const &other)
        {
            iterator(other).swap(*this);
            return *this;
        }

        iterator &operator=(base_type const &other)
        {
            base_type(other).swap(*this);
            return *this;
        }

        iterator operator+(difference_type const &s) const
        {
            iterator res(*this);
            res.advance(s);
            return std::move(res);
        }

        void swap(iterator &other)
        {
            base_type::swap(other);
            std::swap(m_iform_, other.m_iform_);
        }

        id_type operator*() const { return pack_(base_type::operator*()); }

        id_type pack_(nTuple<index_type, ndims + 1> const &idx) const
        {
            return pack_index(idx, m_sub_index_to_id_[m_iform_][idx[ndims]]);
        }

    };

    struct range_type
    {
    private:
        typedef range_type this_type;
    public:

        typedef iterator const_iterator;

        range_type() : m_iform_(VERTEX), m_min_(), m_max_(m_min_), m_grain_size_(m_min_) { }

        // constructors

        range_type(index_tuple const &b, index_tuple const &e, int IFORM = VERTEX)
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
        range_type(T0 const &b, T1 const &e, index_tuple const &grain_size, int IFORM = VERTEX)
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

        // Proportional split is enabled
        static const bool is_splittable_in_proportion = true;

        // capacity


        bool empty() const { return m_min_ == m_max_; }

        size_t size() const
        {
            return ((m_iform_ == VERTEX || m_iform_ == VOLUME) ? 1 : 3) * NProduct(m_max_ - m_min_);
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
        const_iterator begin() const { return const_iterator(m_min_, m_min_, m_max_, m_iform_); }

        const_iterator end() const { return const_iterator(m_min_, m_min_, m_max_, m_iform_).end(); }

    private:


        int m_iform_;
        index_tuple m_min_, m_max_, m_grain_size_;
    };

//    typedef Range<iterator> range_type;

    template<typename T0, typename T1>
    static range_type make_range(T0 const &min, T1 const &max, int iform = VERTEX)
    {
        return range_type(min, max, iform);
    }

    template<int IFORM, typename TB>
    static range_type make_range(TB const &b)
    {
        return make_range(std::get<0>(b), std::get<1>(b), IFORM);
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
                        (unpack_index(s, 2) + e[2] - b[2] - b[2]) % (e[2] - b[2]) +

                        (
                                ((unpack_index(s, 1) + e[1] - b[1] - b[1]) % (e[1] - b[1])) +

                                ((unpack_index(s, 0) + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])

                        ) * (e[2] - b[2])

                ) * num_of_ele_in_cell(s) + sub_index(s);

    }
//
//    static constexpr index_type hash(id_type s, id_type b, id_type e)
//    {
//        return hash(s, unpack_index(b), unpack_index(e));
//
//        //  return hash_((diff(s, b) + diff(e, b)), diff(e, b)) * num_of_ele_in_cell(s) + sub_index(s);
//    }
//
//
//    static constexpr index_type hash_(id_type const &s, id_type const &d)
//    {
//        //C-ORDER SLOW FIRST
//
//        return
//
//                (UNPACK_INDEX(s, 2) % UNPACK_INDEX(d, 2)) +
//
//                (
//                        (UNPACK_INDEX(s, 1) % UNPACK_INDEX(d, 1)) +
//                        (UNPACK_INDEX(s, 0) % UNPACK_INDEX(d, 0)) * UNPACK_INDEX(d, 1)
//                )
//
//                * UNPACK_INDEX(d, 2);
//
//    }

    template<size_t IFORM>
    static constexpr size_t max_hash(id_type b, id_type e)
    {
        return NProduct(unpack_index((e - b)))
               * m_id_to_num_of_ele_in_cell_[sub_index_to_id<IFORM>(0)];
    }

    template<typename TGeometry>
    static void get_element_volume_in_cell(TGeometry const &geo, id_type s0, Real *v, Real *inv_v, Real *dual_v,
                                           Real *inv_dual_v)
    {

        /**
         *\verbatim
         *                ^y
         *               /
         *        z     /
         *        ^    /
         *        |   6---------------7
         *        |  /|              /|
         *        | / |             / |
         *        |/  |            /  |
         *        4---|-----------5   |
         *        |   |           |   |
         *        |   2-----------|---3
         *        |  /            |  /
         *        | /             | /
         *        |/              |/
         *        0---------------1---> x
         *
         *\endverbatim
         */



        typedef typename TGeometry::point_type point_type;

        auto dims = geo.dimensions();


        static constexpr id_type HI = 1UL << (MESH_RESOLUTION);
        static constexpr id_type HJ = HI << ID_DIGITS;
        static constexpr id_type HK = HI << (ID_DIGITS * 2);

        static constexpr id_type HA = HI | HJ | HK;
        //primary
        {
            size_t s = (s0 | FULL_OVERFLOW_FLAG);

            point_type p[NUM_OF_NODE_ID] = {

                    /*000*/  geo.point(s),                       //
                    /*001*/  geo.point(s + (HI)),          //
                    /*010*/  geo.point(s + (HJ)),          //
                    /*011*/  geo.point(s + (HJ | HI)),  //

                    /*100*/  geo.point(s + (HK)),          //
                    /*101*/  geo.point(s + (HK | HI)),   //
                    /*110*/  geo.point(s + (HK | HJ)),   //
                    /*111*/  geo.point(s + (HK | HJ | HI))    //

            };


            v[TAG_VERTEX] = 1;

            v[TAG_EDGE0] = geo.simplex_length(p[0], p[1]);
            v[TAG_EDGE1] = geo.simplex_length(p[0], p[2]);
            v[TAG_EDGE2] = geo.simplex_length(p[0], p[4]);

            v[TAG_FACE0] = geo.simplex_area(p[0], p[2], p[6]) + geo.simplex_area(p[0], p[6], p[4]);
            v[TAG_FACE1] = geo.simplex_area(p[0], p[1], p[5]) + geo.simplex_area(p[0], p[5], p[4]);
            v[TAG_FACE2] = geo.simplex_area(p[0], p[1], p[3]) + geo.simplex_area(p[0], p[3], p[2]);


            v[TAG_VOLUME] = geo.simplex_volume(p[0], p[1], p[2], p[4]) + //
                            geo.simplex_volume(p[1], p[4], p[5], p[2]) + //
                            geo.simplex_volume(p[2], p[6], p[4], p[5]) + //
                            geo.simplex_volume(p[1], p[3], p[2], p[5]) + //
                            geo.simplex_volume(p[3], p[5], p[7], p[6]) + //
                            geo.simplex_volume(p[3], p[6], p[2], p[5]);

        }
        //dual
        {
            size_t s = (s0 | FULL_OVERFLOW_FLAG) - (HA >> 1);

//            point_type p[NUM_OF_NODE_ID] = {
//
//                    /*000*/    geo.point(s + ((LK | LJ | LI) << 1)),   //
//                    /*001*/    geo.point(s + ((LK | LJ | HI) << 1)),   //
//                    /*010*/    geo.point(s + ((LK | HJ | LI) << 1)),   //
//                    /*011*/    geo.point(s + ((LK | HJ | HI) << 1)),   //
//
//                    /*100*/    geo.point(s + ((HK | LJ | LI) << 1)),   //
//                    /*101*/    geo.point(s + ((HK | LJ | HI) << 1)),   //
//                    /*110*/    geo.point(s + ((HK | HJ | LI) << 1)),   //
//                    /*111*/    geo.point(s + ((HK | HJ | HI) << 1))    //
//
//            };

            point_type p[NUM_OF_NODE_ID] = {

                    /*000*/  geo.point(s),                       //
                    /*001*/  geo.point(s + (HI)),          //
                    /*010*/  geo.point(s + (HJ)),          //
                    /*011*/  geo.point(s + (HJ | HI)),  //

                    /*100*/  geo.point(s + (HK)),          //
                    /*101*/  geo.point(s + (HK | HI)),   //
                    /*110*/  geo.point(s + (HK | HJ)),   //
                    /*111*/  geo.point(s + (HK | HJ | HI))    //

            };


            dual_v[TAG_VOLUME] = 1;

            dual_v[TAG_FACE0] = geo.simplex_length(p[6], p[7]);
            dual_v[TAG_FACE1] = geo.simplex_length(p[5], p[7]);
            dual_v[TAG_FACE2] = geo.simplex_length(p[3], p[7]);


            dual_v[TAG_EDGE0] = geo.simplex_area(p[1], p[3], p[5]) + geo.simplex_area(p[3], p[7], p[5]);
            dual_v[TAG_EDGE1] = geo.simplex_area(p[2], p[3], p[7]) + geo.simplex_area(p[2], p[7], p[6]);
            dual_v[TAG_EDGE2] = geo.simplex_area(p[4], p[5], p[7]) + geo.simplex_area(p[4], p[7], p[6]);


            dual_v[TAG_VERTEX] = geo.simplex_volume(p[0], p[1], p[2], p[4]) + //
                                 geo.simplex_volume(p[1], p[4], p[5], p[2]) + //
                                 geo.simplex_volume(p[2], p[6], p[4], p[5]) + //
                                 geo.simplex_volume(p[1], p[3], p[2], p[5]) +
                                 geo.simplex_volume(p[3], p[5], p[7], p[6]) + //
                                 geo.simplex_volume(p[3], p[6], p[2], p[5])  //
                    ;

        }

        for (int i = 0; i < NUM_OF_NODE_ID; ++i)
        {
            inv_v[i] = 1.0 / v[i];
            inv_dual_v[i] = 1.0 / dual_v[i];
        }


        if (dims[0] <= 1)
        {
            inv_v[TAG_EDGE0] = 0;
            inv_dual_v[TAG_FACE0] = 0;
        }

        if (dims[1] <= 1)
        {
            inv_v[TAG_EDGE1] = 0;
            inv_dual_v[TAG_FACE1] = 0;
        }

        if (dims[2] <= 1)
        {
            inv_v[TAG_EDGE2] = 0;

            inv_dual_v[TAG_FACE2] = 0;
        }


    }
};


/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

constexpr int MeshIDs::ndims;
constexpr int MeshIDs::MESH_RESOLUTION;
constexpr Real MeshIDs::EPSILON;

constexpr int MeshIDs::FULL_DIGITS;
constexpr int MeshIDs::ID_DIGITS;


constexpr typename MeshIDs::id_type MeshIDs::OVERFLOW_FLAG;
constexpr typename MeshIDs::id_type MeshIDs::ID_ZERO;
constexpr typename MeshIDs::id_type MeshIDs::INDEX_ZERO;


constexpr typename MeshIDs::id_type MeshIDs::ID_MASK;
constexpr typename MeshIDs::id_type MeshIDs::_DK;
constexpr typename MeshIDs::id_type MeshIDs::_DJ;
constexpr typename MeshIDs::id_type MeshIDs::_DI;
constexpr typename MeshIDs::id_type MeshIDs::_DA;


constexpr int MeshIDs::m_id_to_index_[];

constexpr int MeshIDs::m_id_to_iform_[];

constexpr int MeshIDs::m_id_to_num_of_ele_in_cell_[];

constexpr int MeshIDs::m_adjacent_cell_num_[4][8];


constexpr typename MeshIDs::id_type MeshIDs::m_id_to_shift_[];
constexpr int MeshIDs::m_sub_index_to_id_[4][3];


constexpr typename MeshIDs::id_type
        MeshIDs::m_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/];

constexpr typename MeshIDs::coordinates_tuple MeshIDs::m_id_to_coordinates_shift_[];

}//namespace  mesh
}// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

