//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_SP_DEF_H
#define SIMPLA_SP_DEF_H

#include <simpla/SIMPLA_config.h>
#include <cassert>
#include <limits>
#include <string>
namespace simpla {

// enum POSITION
//{
//	/*
//	 FULL = -1, // 11111111
//	 CENTER = 0, // 00000000
//	 LEFT = 1, // 00000001
//	 RIGHT = 2, // 00000010
//	 DOWN = 4, // 00000100
//	 UP = 8, // 00001000
//	 BACK = 16, // 00010000
//	 FRONT = 32 //00100000
//	 */
//	FULL = -1, //!< FULL
//	CENTER = 0, //!< CENTER
//	LEFT = 1,  //!< LEFT
//	RIGHT = 2, //!< RIGHT
//	DOWN = 4,  //!< DOWN
//	UP = 8,    //!< UP
//	BACK = 16, //!< BACK
//	FRONT = 32 //!< FRONT
//};
//
enum ArrayOrder {
    C_ORDER,       // SLOW FIRST
    FORTRAN_ORDER  //  FAST_FIRST
};

typedef Real scalar_type;

static constexpr Real INIFITY = std::numeric_limits<Real>::infinity();

static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();
}




#endif  // SIMPLA_SP_DEF_H
