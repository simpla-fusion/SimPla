/**
 * sp_def.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_DEF_LITE_H_
#define SP_DEF_LITE_H_

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include "../src/sp_cwrap.h"

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"

typedef int8_t byte_type; // int8_t
typedef float Real;
typedef int Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;

#define SP_SUCCESS 0
#define SP_FAILED  1
#define DONE        printf( "====== DONE ======\n" );
#define CHECK        printf( "[ line %d in file%s]====== CHECK ======\n", __LINE__, __FILE__ );

#endif /* SP_DEF_LITE_H_ */
