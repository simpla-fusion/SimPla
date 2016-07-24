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
#include "../src/sp_config.h"

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"


#define DONE        printf( "====== DONE ======\n" );
#define CHECK(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_) );
#define ERROR(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__, _MSG_);exit(-1);

#define CHECK_INT(_MSG_)    printf( "%s:%d:0:%s: %s = 0x%lx \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(long)(_MSG_) );
#define CHECK_STR(_MSG_)    printf( "%s:%d:0:%s: %s = %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_), (_MSG_) );

typedef MeshEntityId64 MeshEntityId;

typedef MeshEntityId32 MeshEntityShortId;


#endif /* SP_DEF_LITE_H_ */
