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


#define DONE       if(spMPIRank()==0){ printf( "====== DONE ======\n" );}
#define CHECK(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_) );
#define ERROR(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__, _MSG_);exit(-1);
#define UNIMPLEMENTED       printf( "%s:%d:0:%s: UNIMPLEMENTED!! \n", __FILE__, __LINE__,__PRETTY_FUNCTION__ );

#define CHECK_REAL(_MSG_)    printf( "%s:%d:0:%s: %s =%e \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(Real)(_MSG_) );
#define CHECK_INT(_MSG_)    printf( "%s:%d:0:%s: %s = %ld \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(long)(_MSG_) );
#define CHECK_STR(_MSG_)    printf( "%s:%d:0:%s: %s = %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_), (_MSG_) );

#define SP_CALL(_CMD_)                                                                                        \
{                                                                                                                    \
   int _return_code=_CMD_;                                                                                            \
   if(_return_code==SP_FAILED)                                                                                         \
   {                                                                                                                  \
        printf( "%s:%d:0:%s: command failed! [%s] \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_CMD_));  \
   }                                                                                                                 \
}


typedef MeshEntityId64 MeshEntityId;

typedef MeshEntityId32 MeshEntityShortId;

#define TWOPI (3.141592653589793f*2.0f)


#define MIN(_A_, _B_) (_A_<_B_)?_A_:_B_

#define MAX(_A_, _B_) (_A_>_B_)?_A_:


#ifdef USING_DEVICE_MEMORY
#   undef USING_DEVICE_MEMORY;
#endif


#endif /* SP_DEF_LITE_H_ */
