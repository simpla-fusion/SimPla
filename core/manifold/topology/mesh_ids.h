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
#include "../../gtl/iterator/range.h"

#include "../manifold_traits.h"

namespace simpla
{

template<size_t TAGS>
struct MeshIDs_;


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
template<size_t TAGS = 4>
struct MeshIDs_
{
    /// @name level independent
    /// @{

    static constexpr int ndims = 3;
    static constexpr int MESH_RESOLUTION = static_cast<int> (TAGS & 0xF);

    typedef MeshIDs_<TAGS> this_type;

    typedef std::uint64_t id_type;

    typedef nTuple<id_type, ndims> id_tuple;

    typedef nTuple<Real, ndims> coordinates_tuple;

    typedef long index_type;

    typedef nTuple<index_type, ndims> index_tuple;

//	typedef Real coordinate_type;

    static constexpr int FULL_DIGITS = std::numeric_limits<id_type>::digits;

    static constexpr id_type ID_DIGITS = 21;

    static constexpr id_type HEAD_DIGITS = (FULL_DIGITS - ID_DIGITS * 3);

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

    static constexpr id_type PRIMARY_ID_MASK_ = ID_MASK & (~SUB_ID_MASK);

    static constexpr id_type PRIMARY_ID_MASK = PRIMARY_ID_MASK_
                                               | (PRIMARY_ID_MASK_ << ID_DIGITS)
                                               | (PRIMARY_ID_MASK_ << (ID_DIGITS * 2));

    static constexpr id_type _DA = _DI | _DJ | _DK;

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

            {0,  0,  0},            // 000
            {_R, 0,  0},           // 001
            {0,  _R, 0},           // 010
            {0,  0,  _R},           // 011
            {_R, _R, 0},          // 100
            {_R, 0,  _R},          // 101
            {0,  _R, _R},          // 110
            {0,  _R, _R},          // 111

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

    static constexpr id_type diff(id_type a, id_type b)
    {
        return ((a | OVERFLOW_FLAG) - b) & (~OVERFLOW_FLAG);
    }
//
//    static constexpr id_type minus(id_type a, id_type b)
//    {
//        rezturn ((a | OVERFLOW_FLAG) - b) & (~OVERFLOW_FLAG);
//    }
//
//    static constexpr id_type plus(id_type a, id_type b)
//    {
//        return (a + b) & (~OVERFLOW_FLAG);
//    }

#define UNPACK_ID(_S_, _I_)    (static_cast<id_type>(_S_) >> (ID_DIGITS*(_I_ )) & ID_MASK)
#define UNPACK_INDEX(_S_, _I_)  (static_cast<index_type>(((_S_)>> (ID_DIGITS*(_I_ )) & ID_MASK)>>MESH_RESOLUTION)-static_cast<index_type>(OVERFLOW_FLAG>>MESH_RESOLUTION))

    template<typename T>
    static constexpr id_type pack(T const &idx)
    {
        return (static_cast<id_type>(idx[0]) & ID_MASK)
               | ((static_cast<id_type>(idx[1]) & ID_MASK) << ID_DIGITS)
               | ((static_cast<id_type>(idx[2]) & ID_MASK) << (ID_DIGITS * 2));
    }

    static constexpr id_type pack(id_type i0, id_type i1, id_type i2)
    {
        return (i0 & ID_MASK) | ((i1 & ID_MASK) << ID_DIGITS) | ((i2 & ID_MASK) << (ID_DIGITS * 2));
    }

    static constexpr id_tuple unpack(id_type s)
    {
        return id_tuple({UNPACK_ID(s, 0), UNPACK_ID(s, 1), UNPACK_ID(s, 2)});;
    }

    static constexpr index_tuple unpack_index(id_type s)
    {
        return index_tuple({UNPACK_INDEX(s, 0), UNPACK_INDEX(s, 1), UNPACK_INDEX(s, 2)});
    }

    template<typename T>
    static constexpr id_type pack_index(T const &idx, int n_id = 0)
    {
        return (pack(idx) << MESH_RESOLUTION) | FULL_OVERFLOW_FLAG | m_id_to_shift_[n_id];
    }


    template<typename T>
    static constexpr T type_cast(id_type s)
    {
        return static_cast<T>(unpack(s));
    }

//    static coordinates_tuple coordinates(id_type s)
//    {
//        return coordinates_tuple{static_cast<Real>(UNPACK_ID(s, 0) - OVERFLOW_FLAG) * MESH_COORDINATES_FACTOR,
//                                 static_cast<Real>(UNPACK_ID(s, 1) - OVERFLOW_FLAG) * MESH_COORDINATES_FACTOR,
//                                 static_cast<Real>(UNPACK_ID(s, 2) - OVERFLOW_FLAG) * MESH_COORDINATES_FACTOR};
//    }

    static coordinates_tuple coordinates(id_type s)
    {
        return coordinates_tuple{static_cast<Real>(UNPACK_ID(s, 0)),
                                 static_cast<Real>(UNPACK_ID(s, 1)),
                                 static_cast<Real>(UNPACK_ID(s, 2))};
    }

    static coordinates_tuple point(nTuple<index_type, ndims> const &idx)
    {
        return coordinates_tuple{
                static_cast<Real>((idx[0] << MESH_RESOLUTION) + OVERFLOW_FLAG),
                static_cast<Real>((idx[1] << MESH_RESOLUTION) + OVERFLOW_FLAG),
                static_cast<Real>((idx[2] << MESH_RESOLUTION) + OVERFLOW_FLAG)
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

    static constexpr coordinates_tuple coordinates_local_to_global(
            std::tuple<id_type, coordinates_tuple> const &t)
    {
        return coordinates(std::get<0>(t)) + std::get<1>(t);
    }

//! @name id auxiliary functions
//! @{
    static constexpr id_type dual(id_type s)
    {
        return (s & (~_DA)) | ((~(s & _DA)) & _DA);

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
     *        |  q7--------------q6
     *        |  /|              /|
     *          / |             / |
     *         /  |            /  |
     *       q4---|----------q5   |
     *        |   |     x0    |   |
     *        |  q3-----------|--q2
     *        |  /            |  /
     *        | /             | /
     *        |/              |/
     *       q0--------------q1   ---> x
     *
     *   \endverbatim
     */
    static constexpr int MAX_NUM_OF_CELL = 12;

    static constexpr id_type _HI = _D;
    static constexpr id_type _HJ = _HI << ID_DIGITS;
    static constexpr id_type _HK = _HI << (ID_DIGITS * 2);
    static constexpr id_type _LI = (-_D) & (ID_MASK >> 1);
    static constexpr id_type _LJ = _LI << ID_DIGITS;
    static constexpr id_type _LK = _LI << (ID_DIGITS * 2);

    static constexpr int m_adjoint_num_[4/* to iform*/][8/* node id*/] =

            { // VERTEX
                    {
                            /* 000*/1,
                            /* 001*/2,
                            /* 010*/2,
                            /* 011*/4,
                            /* 100*/2,
                            /* 101*/4,
                            /* 110*/4,
                            /* 111*/8},

                    // EDGE
                    {
                            /* 000*/6,
                            /* 001*/1,
                            /* 010*/1,
                            /* 011*/4,
                            /* 100*/1,
                            /* 101*/4,
                            /* 110*/4,
                            /* 111*/12},

                    // FACE
                    {
                            /* 000*/12,
                            /* 001*/4,
                            /* 010*/4,
                            /* 011*/1,
                            /* 100*/4,
                            /* 101*/1,
                            /* 110*/1,
                            /* 111*/6},

                    // VOLUME
                    {
                            /* 000*/8,
                            /* 001*/4,
                            /* 010*/4,
                            /* 011*/2,
                            /* 100*/4,
                            /* 101*/2,
                            /* 110*/2,
                            /* 111*/1}

            };

    static constexpr id_type m_adjoint_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/] =
            {
                    //To VERTEX
                    {

                            /* 000*/
                            {0},
                            /* 001*/
                            {_LI,       _HI},
                            /* 010*/
                            {_LJ,       _HJ},
                            /* 011*/
                            {_LI | _LJ, _HI | _LJ, _HI | _HJ, _LI | _HJ},
                            /* 100*/
                            {_LK,       _HK},
                            /* 101*/
                            {_LK | _LI, _HK | _LI, _HK | _HI, _LK | _HI},
                            /* 110*/
                            {_LJ | _LK, _HJ | _LK, _HJ | _HK, _LJ | _HK},
                            /* 111*/
                            {_LI | _LJ | _LK, //
                                    _HI | _LJ | _LK, //
                                         _HI | _HJ | _LK, //
                                              _LI | _HJ | _LK, //

                                                   _LI | _LJ | _HK, //
                                                        _HI | _LJ | _HK, //
                                    _HI | _HJ | _HK, //
                                    _LI | _HJ | _HK}

                    },

                    //To EDGE
                    {
                            /* 000*/
                            {_HI, _LI, _HJ, _LJ, _HK, _LK},
                            /* 001*/
                            {0},
                            /* 010*/
                            {0},
                            /* 011*/
                            {_LJ,       _HI,       _HJ,       _LI},
                            /* 100*/
                            {0},
                            /* 101*/
                            {_LI,       _HK,       _HI,       _LK},
                            /* 110*/
                            {_LK,       _HJ,       _HK,       _LJ},
                            /* 111*/
                            {_LK | _LJ,  //-> 001
                                    _LK | _HI,  //   012
                                         _LK | _HJ,  //   021
                                              _LK | _LI,  //   010

                                                   _LI | _LJ,  //
                                                        _LI | _HJ,  //
                                    _HI | _LJ,  //
                                    _HI | _HJ,  //

                                    _HK | _LJ,  //
                                    _HK | _HI,  //
                                    _HK | _HJ,  //
                                    _HK | _LI  //
                            }},

                    //To FACE
                    {
                            /* 000*/
                            {_LK | _LJ,  //
                                  _LK | _HI,  //
                                       _LK | _HJ,  //
                                            _LK | _LI,  //

                                                 _LI | _LJ,  //
                                                      _LI | _HJ,  //
                                    _HI | _LJ,  //
                                    _HI | _HJ,  //

                                    _HK | _LJ,  //
                                    _HK | _HI,  //
                                    _HK | _HJ,  //
                                    _HK | _LI  //
                            },
                            /* 001*/
                            {_LJ,       _HK,       _HJ,       _LK},
                            /* 010*/
                            {_LK,       _HI,       _HK,       _LI},
                            /* 011*/
                            {0},
                            /* 100*/
                            {_LI,       _HJ,       _HI,       _LJ},
                            /* 101*/
                            {0},
                            /* 110*/
                            {0},
                            /* 111*/
                            {_LI,   _LJ, _LK, _HI, _HJ, _HK}},
                    // TO VOLUME
                    {
                            /* 000*/
                            {_LI | _LJ | _LK,  //
                                  _LI | _HJ | _LK,  //
                                       _LI | _LJ | _HK,  //
                                            _LI | _HJ | _HK,  //

                                                 _HI | _LJ | _LK,  //
                                                      _HI | _HJ | _LK,  //
                                    _HI | _LJ | _HK,  //
                                    _HI | _HJ | _HK  //

                            },
                            /* 001*/
                            {_LJ | _LK, _LJ | _HK, _HJ | _LK, _HJ | _HK},
                            /* 010*/
                            {_LK | _LI, _LK | _HI, _HK | _LI, _HK | _HI},
                            /* 011*/
                            {_LK,       _HK},
                            /* 100*/
                            {_LI | _LJ, _LI | _HJ, _HI | _LJ, _HI | _HJ},
                            /* 101*/
                            {_LJ,       _HJ},
                            /* 110*/
                            {_LI,       _HI},
                            /* 111*/
                            {0}}

            };

    static int get_adjoints(id_type s, size_t IFORM, size_t nodeid,
                            id_type *res = nullptr)
    {
        if (res != nullptr)
        {
            for (int i = 0; i < m_adjoint_num_[IFORM][nodeid]; ++i)
            {
                res[i] = ((s + m_adjoint_matrix_[IFORM][nodeid][i]));
            }
        }
        return m_adjoint_num_[IFORM][nodeid];
    }

    template<size_t IFORM, size_t NODE_ID>
    static int get_adjoints(id_type s, id_type *res = nullptr)
    {
        return get_adjoints(s, IFORM, NODE_ID, res);
    }

    template<size_t AXE>
    static constexpr std::tuple<id_type, id_type> primary_line(id_type s)
    {
        return std::make_tuple(((s | OVERFLOW_FLAG) - (_D << (ID_DIGITS * AXE))) & (~OVERFLOW),
                               s + (_D << (ID_DIGITS * AXE)));
    }

    template<size_t AXE>
    static constexpr std::tuple<id_type, id_type> pixel(id_type s)
    {
        return std::make_tuple(((s | OVERFLOW_FLAG) - (_DA & (~(_D << (ID_DIGITS * AXE))))) & (~OVERFLOW),
                               s + (_DA & (~(_D << (ID_DIGITS * AXE)))));
    }

    static constexpr std::tuple<id_type, id_type> voxel(id_type s)
    {
        return std::make_tuple(((s | OVERFLOW_FLAG) - _DA) & (~OVERFLOW), s + _DA);
    }

    template<typename TID>
    static TID bit_shift_id(TID s, size_t n)
    {
        id_type m = (1UL << (ID_DIGITS - n - 1)) - 1;
        return ((s & (m | (m << ID_DIGITS) | (m << (ID_DIGITS * 2)))) << n) & (~FULL_OVERFLOW_FLAG);
    }

//    static constexpr id_type id_add(id_type s, id_type d)
//    {
//        return ((s & (~FULL_OVERFLOW_FLAG)) + d) & (~FULL_OVERFLOW_FLAG);
//    }
//
//    static constexpr id_type id_minus(id_type s, id_type d)
//    {
//        return ((s & (~FULL_OVERFLOW_FLAG)) - d) & (~FULL_OVERFLOW_FLAG);
//    }


    struct iterator : public block_iterator<index_type, ndims + 1>
    {
    private:
        typedef block_iterator<index_type, ndims + 1> base_type;

        int m_iform_;

    public:

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
                base_type(pself, pmin, pmax), m_iform_(IFORM)
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

        void swap(iterator &other)
        {
            base_type::swap(other);
            std::swap(m_iform_, other.m_iform_);
        }

        constexpr id_type operator*() const { return pack_(base_type::operator*()); }

        constexpr id_type pack_(nTuple<index_type, ndims + 1> const &idx) const
        {
            return (pack(idx) << MESH_RESOLUTION) | FULL_OVERFLOW_FLAG |
                   m_id_to_shift_[m_sub_index_to_id_[m_iform_][idx[ndims]]];
        }

    };


    typedef Range<iterator> range_type;

    template<typename T0, typename T1>
    static range_type range(T0 const &min, T1 const &max, int iform = VERTEX)
    {
        iterator ib(min, min, max, iform);

        return range_type(ib, ib.end());
    }

    template<int IFORM, typename T0, typename T1>
    static range_type range(T0 const &b, T1 const &e)
    {
        return range(b, e, IFORM);
    }

    static range_type range(id_type const &b, id_type const &e)
    {
        return range_type(iterator(b, b, e), iterator(e, b, e));
    }


    static constexpr index_type hash(id_type const &s, index_tuple const &b, index_tuple const &e)
    {
        //C-ORDER SLOW FIRST

        return
                (
                        (UNPACK_INDEX(s, 2) + e[2] - b[2] - b[2]) % (e[2] - b[2]) +

                        (
                                ((UNPACK_INDEX(s, 1) + e[1] - b[1] - b[1]) % (e[1] - b[1])) +

                                ((UNPACK_INDEX(s, 0) + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])

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
        return NProduct(unpack_index(diff(e, b)))
               * m_id_to_num_of_ele_in_cell_[sub_index_to_id<IFORM>(0)];
    }


};


namespace traits
{
template<typename>
struct id_type;
template<size_t TAGS>
struct id_type<MeshIDs_<TAGS>>
{
    typedef typename MeshIDs_<TAGS>::id_type type;
};

template<typename>
struct coordinates_tuple;
template<size_t TAGS>
struct coordinates_tuple<MeshIDs_<TAGS>>
{
    typedef typename MeshIDs_<TAGS>::coordinates_tuple type;
};

}  // namespace traits
/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<size_t TAGS> constexpr int MeshIDs_<TAGS>::ndims;
template<size_t TAGS> constexpr int MeshIDs_<TAGS>::MESH_RESOLUTION;
template<size_t TAGS> constexpr Real MeshIDs_<TAGS>::EPSILON;

template<size_t TAGS> constexpr int MeshIDs_<TAGS>::FULL_DIGITS;


template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::OVERFLOW_FLAG;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::ID_ZERO;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::INDEX_ZERO;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::ID_DIGITS;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::ID_MASK;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::_DK;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::_DJ;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::_DI;
template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::_DA;


template<size_t TAGS> constexpr int MeshIDs_<TAGS>::m_id_to_index_[];

template<size_t TAGS> constexpr int MeshIDs_<TAGS>::m_id_to_iform_[];

template<size_t TAGS> constexpr int MeshIDs_<TAGS>::m_id_to_num_of_ele_in_cell_[];

template<size_t TAGS> constexpr int MeshIDs_<TAGS>::m_adjoint_num_[4][8];


template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type MeshIDs_<TAGS>::m_id_to_shift_[];
template<size_t TAGS> constexpr int MeshIDs_<TAGS>::m_sub_index_to_id_[4][3];


template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::id_type
        MeshIDs_<TAGS>::m_adjoint_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/];

template<size_t TAGS> constexpr typename MeshIDs_<TAGS>::coordinates_tuple MeshIDs_<TAGS>::m_id_to_coordinates_shift_[];

typedef MeshIDs_<> MeshIDs;

}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

