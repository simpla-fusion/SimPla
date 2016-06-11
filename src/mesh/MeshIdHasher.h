//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_MESHIDHASHER_H
#define SIMPLA_MESHIDHASHER_H


#include "../sp_config.h"


#ifdef __cplusplus
extern "C" {
#endif



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

#define SP_STATIC_CONST   static  const
/// @name level independent
/// @{

SP_STATIC_CONST int ndims = 3;
SP_STATIC_CONST int MAX_NUM_OF_NEIGHBOURS = 12;
SP_STATIC_CONST int MESH_RESOLUTION = 1;


SP_STATIC_CONST int FULL_DIGITS = 64; // id_type is int64_t

SP_STATIC_CONST int ID_DIGITS = 21;

SP_STATIC_CONST int HEAD_DIGITS = (FULL_DIGITS - ID_DIGITS * 3);

SP_STATIC_CONST id_type ID_MASK = (1UL << ID_DIGITS) - 1;

SP_STATIC_CONST id_type NO_HAED = (1UL << (ID_DIGITS * 3)) - 1;

SP_STATIC_CONST id_type OVERFLOW_FLAG = (1UL) << (ID_DIGITS - 1);

SP_STATIC_CONST id_type FULL_OVERFLOW_FLAG =
        OVERFLOW_FLAG | (OVERFLOW_FLAG << ID_DIGITS) | (OVERFLOW_FLAG << (ID_DIGITS * 2));

SP_STATIC_CONST id_type INDEX_ZERO = (1UL) << (ID_DIGITS - 2);

SP_STATIC_CONST id_type ID_ZERO = INDEX_ZERO | (INDEX_ZERO << ID_DIGITS) | (INDEX_ZERO << (ID_DIGITS * 2));

SP_STATIC_CONST Real EPSILON = 1.0 / (Real) (INDEX_ZERO);


/// @}

/// @name level dependent
/// @{

SP_STATIC_CONST id_type SUB_ID_MASK = ((1UL << MESH_RESOLUTION) - 1);

SP_STATIC_CONST id_type _D = 1UL << (MESH_RESOLUTION - 1);

SP_STATIC_CONST Real _R = (Real) (_D);


SP_STATIC_CONST id_type _DI = _D;
SP_STATIC_CONST id_type _DJ = _D << (ID_DIGITS);
SP_STATIC_CONST id_type _DK = _D << (ID_DIGITS * 2);
SP_STATIC_CONST id_type _DA = _DI | _DJ | _DK;


SP_STATIC_CONST id_type PRIMARY_ID_MASK_ = ID_MASK & (~SUB_ID_MASK);
SP_STATIC_CONST id_type PRIMARY_ID_MASK = PRIMARY_ID_MASK_
                                          | (PRIMARY_ID_MASK_ << ID_DIGITS)
                                          | (PRIMARY_ID_MASK_ << (ID_DIGITS * 2));


SP_STATIC_CONST Real GRID_WIDTH = (Real) (1UL << MESH_RESOLUTION);
SP_STATIC_CONST Real INV_GRID_WIDTH = 1.0 / GRID_WIDTH;

/// @}

SP_STATIC_CONST int spm_sub_index_to_id_[4][3] = { //

        {0, 0, 0}, /*VERTEX*/
        {1, 2, 4}, /*EDGE*/
        {6, 5, 3}, /*FACE*/
        {7, 7, 7} /*VOLUME*/

};

SP_STATIC_CONST id_type spm_id_to_sub_index_[8] = { //

        0, // 000
        0, // 001
        1, // 010
        2, // 011
        2, // 100
        1, // 101
        0, // 110
        0, // 111
};

SP_STATIC_CONST id_type spm_id_to_shift_[] = {

        0,                    // 000
        _DI,                    // 001
        _DJ,                    // 010
        (_DI | _DJ),                    // 011
        _DK,                    // 100
        (_DK | _DI),                    // 101
        (_DJ | _DK),                    // 110
        _DA                    // 111

};

SP_STATIC_CONST Real spm_id_to_coordinates_shift_[][3] = {

        {0,  0,  0},            // 000
        {_R, 0,  0},           // 001
        {0,  _R, 0},           // 010
        {0,  0,  _R},           // 011
        {_R, _R, 0},          // 100
        {_R, 0,  _R},          // 101
        {0,  _R, _R},          // 110
        {0,  _R, _R},          // 111

};
SP_STATIC_CONST int spm_iform_to_num_of_ele_in_cell_[] = {
        1, // VETEX
        3, // EDGE
        3, // FACE
        1  // VOLUME
};
SP_STATIC_CONST int spm_id_to_num_of_ele_in_cell_[] = {

        1,        // 000
        3,        // 001
        3,        // 010
        3,        // 011
        3,        // 100
        3,        // 101
        3,        // 110
        1        // 111
};

SP_STATIC_CONST int spm_id_to_iform_[] = { //

        0, // 000
        1, // 001
        1, // 010
        2, // 011
        1, // 100
        2, // 101
        2, // 110
        3 // 111
};

static inline size_type sp_node_id(id_type s);

static inline id_type sp_mininal_vertex(id_type s)
{
    return (s | FULL_OVERFLOW_FLAG) - (_DA);

}

static inline int sp_sub_index_to_id(int IFORM, int n)
{
    return spm_sub_index_to_id_[IFORM][n];
}

static inline int sp_iform(id_type s) { return spm_id_to_iform_[sp_node_id(s)]; }

static inline id_type sp_pack(id_type i0, id_type i1, id_type i2)
{
    return (i0 & ID_MASK) | ((i1 & ID_MASK) << ID_DIGITS) | ((i2 & ID_MASK) << (ID_DIGITS * 2)) |
           FULL_OVERFLOW_FLAG;
}

static inline id_type sp_pack_index_v(index_type const *idx, int n_id)
{
    return sp_pack((id_type) (idx[0]) << MESH_RESOLUTION,

                   (id_type) (idx[1]) << MESH_RESOLUTION,

                   (id_type) (idx[2]) << MESH_RESOLUTION) | spm_id_to_shift_[n_id];
}

static inline id_type sp_pack_index(index_type i, index_type j, index_type k, index_type n_id)
{

    return
            sp_pack((id_type) (i) << MESH_RESOLUTION, (id_type) (j) << MESH_RESOLUTION,
                    (id_type) (k) << MESH_RESOLUTION) | spm_id_to_shift_[n_id];
}

static inline id_type sp_extent_flag_bit(id_type s, int n)
{
    return s | (((s & (1UL << n)) == 0) ? 0UL : ((id_type) (-1L << (n + 1))));
}

static inline id_type sp_unpack_id(id_type s, int n)
{
    return sp_extent_flag_bit(((s & (~FULL_OVERFLOW_FLAG)) >> (ID_DIGITS * n)) & ID_MASK, n);
}

static inline index_type sp_unpack_index(id_type s, int n)
{
    return
            (index_type) (sp_extent_flag_bit(
                    (((s & (~FULL_OVERFLOW_FLAG)) >> (ID_DIGITS * n)) & ID_MASK) >> MESH_RESOLUTION,
                    ID_DIGITS - 2 - MESH_RESOLUTION));
}


static inline void sp_unpack(id_type s, id_type res[3])
{
    res[0] = sp_unpack_id(s, 0);
    res[1] = sp_unpack_id(s, 1);
    res[2] = sp_unpack_id(s, 2);
}

static inline void sp_unpack_indexN(id_type s, index_type res[3])
{

    res[0] = sp_unpack_index(s, 0);
    res[1] = sp_unpack_index(s, 1);
    res[2] = sp_unpack_index(s, 2);
}


static void sp_point(id_type s, Real res[3])
{
    res[0] = (Real) (sp_unpack_index(s, 0));
    res[1] = (Real) (sp_unpack_index(s, 1));
    res[2] = (Real) (sp_unpack_index(s, 2));

}


static inline int sp_num_of_ele_in_cell(id_type s)
{
    return spm_id_to_num_of_ele_in_cell_[sp_node_id(s)];
}


static inline id_type sp_coordinates_global_to_local(Real x[3], int n_id)
{

    id_type s = (sp_pack(x[0] - spm_id_to_coordinates_shift_[n_id][0],
                         x[1] - spm_id_to_coordinates_shift_[n_id][1],
                         x[2] - spm_id_to_coordinates_shift_[n_id][2])
                 & PRIMARY_ID_MASK) | spm_id_to_shift_[n_id];

    Real r[3];

//    r = (x - sp_point(s)) / (_R * 2.0);

    return s;
}

static inline void coordinates_local_to_global(id_type s, Real r[3])
{
    sp_point(s, r);
//    +x * _R * 2;
}


//! @name id auxiliary functions
//! @{
static inline id_type sp_dual(id_type s) { return (s & (~_DA)) | ((~(s & _DA)) & _DA); }

static inline id_type sp_DI(int n, id_type s) { return (s >> (n * ID_DIGITS)) & _D; }

static inline id_type sp_delta_index(id_type s) { return (s & _DA); }

static inline id_type sp_rotate(id_type s)
{
    return ((s & (~_DA)) | (((s & (_DA)) << ID_DIGITS) | ((s & _DK) >> (ID_DIGITS * 2)))) & NO_HAED;
}

static inline id_type sp_inverse_rotate(id_type s)
{
    return ((s & (~_DA)) | (((s & (_DA)) >> ID_DIGITS) | ((s & _DI) << (ID_DIGITS * 2)))) & NO_HAED;
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

#define NUM_OF_NODE_ID  8
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

static inline size_type sp_node_id(id_type s)
{
    return ((s >> (MESH_RESOLUTION - 1)) & 1UL)
           | ((s >> (ID_DIGITS + MESH_RESOLUTION - 2)) & 2UL)
           | ((s >> (ID_DIGITS * 2 + MESH_RESOLUTION - 3)) & 4UL);
}

SP_STATIC_CONST int spm_id_to_index_[8] = { //

        0, // 000
        0, // 001
        1, // 010
        2, // 011
        2, // 100
        1, // 101
        0, // 110
        0, // 111
};

SP_STATIC_CONST int sp_sub_index(id_type s) { return spm_id_to_index_[sp_node_id(s)]; }

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
#define MAX_NUM_OF_ADJACENT_CELL   12


SP_STATIC_CONST int spm_adjacent_cell_num_[4/* to iform*/][8/* node id*/] =

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

SP_STATIC_CONST id_type spm_adjacent_cell_matrix_[4/* to iform*/][NUM_OF_NODE_ID/* node id*/][MAX_NUM_OF_ADJACENT_CELL/*id shift*/] =
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
                        {       _DA},
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


static int sp_get_adjacent_entities(int IFORM, int nodeid, id_type s, id_type *res)
{
    if (res != 0x0)
    {
        for (int i = 0; i < spm_adjacent_cell_num_[IFORM][nodeid]; ++i)
        {
            res[i] = (((s | FULL_OVERFLOW_FLAG) - _DA + spm_adjacent_cell_matrix_[IFORM][nodeid][i])) |
                     (FULL_OVERFLOW_FLAG);
        }
    }
    return spm_adjacent_cell_num_[IFORM][nodeid];
}


static index_type sp_hash(id_type s, index_type const *b, index_type const *e)
{
//C-ORDER SLOW FIRST

    return
            ((sp_unpack_index(s, 2) + e[2] - b[2] - b[2]) % (e[2] - b[2]) +
             (((sp_unpack_index(s, 1) + e[1] - b[1] - b[1]) % (e[1] - b[1])) +
              ((sp_unpack_index(s, 0) + e[0] - b[0] - b[0]) % (e[0] - b[0])) * (e[1] - b[1])) * (e[2] - b[2])) *
            sp_num_of_ele_in_cell(s) + sp_sub_index(s);

}


SP_STATIC_CONST size_type sp_max_hash(index_type const *b, index_type const *e, int IFORM)
{
    return ((e[0] - b[0]) * (e[1] - b[1]) * (e[2] - b[2])) *
           spm_id_to_num_of_ele_in_cell_[spm_sub_index_to_id_[IFORM][0]];
}

/**
 * Solve problem: Undefined reference to SP_FUNCTION_PREFIX char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */


#ifdef __cplusplus
};
#endif


#endif //SIMPLA_MESHIDHASHER_H
