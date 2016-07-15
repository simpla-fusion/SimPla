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
#include "../src/sp_capi.h"

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
#define CHECK(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_) );
#define CHECK_INT(_MSG_)    printf( "%s:%d:0:%s: %s = %d \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(int)(_MSG_) );

#endif /* SP_DEF_LITE_H_ */
