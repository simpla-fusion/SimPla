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
#include "spMPI.h"

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"


#define DONE       if(spMPIRank()==0){ printf( "====== DONE ======\n" );}
#define CHECK(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_) );
#define ERROR(_MSG_)        printf( "%s:%d:0:%s: %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__, _MSG_);exit(-1);
#define UNIMPLEMENTED       printf( "%s:%d:0:%s: UNIMPLEMENTED!! \n", __FILE__, __LINE__,__PRETTY_FUNCTION__ );

#define CHECK_REAL(_MSG_)    printf( "%s:%d:0:%s: %s =%e \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(Real)(_MSG_) );
#define CHECK_INT(_MSG_)    printf( "[%d/%d]%s:%d:0:%s: %s = %ld \n",spMPIRank(),spMPISize(), __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(long)(_MSG_) );
#define CHECK_UINT(_MSG_)    printf( "%s:%d:0:%s: %s = 0x%x \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_),(unsigned long)(_MSG_) );
#define CHECK_STR(_MSG_)    printf( "%s:%d:0:%s: %s = %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_MSG_), (_MSG_) );


//{                                                                                                                    \
//   int _return_code=_CMD_;                                                                                            \
//   if(_return_code==SP_FAILED)                                                                                         \
//   {                                                                                                                  \
//        printf( "%s:%d:0:%s: command failed! [%s] \n", __FILE__, __LINE__,__PRETTY_FUNCTION__,__STRING(_CMD_));  \
//    exit(1); \
//}}
inline int print_error(int error_code, char const *file, int line, char const *function, char const *cmd)
{
    if (error_code == SP_FAILED)
    {
        printf("%s:%d:0:%s: %s: [%s] \n", file, line, function,
               ((error_code == SP_FAILED) ? "FAILED"
                                          : "SUCCESS"),
               cmd);
    }
    return error_code;
}

#define SP_CALL(_CMD_) {if(SP_SUCCESS!= print_error((_CMD_), __FILE__, __LINE__, __PRETTY_FUNCTION__, __STRING(_CMD_))) {return SP_FAILED; }}


typedef MeshEntityId64 MeshEntityId;

typedef MeshEntityId32 MeshEntityShortId;

#define TWOPI (3.141592653589793f*2.0f)


#define MIN(_A_, _B_) (_A_<_B_)?_A_:_B_

#define MAX(_A_, _B_) (_A_>_B_)?_A_:


#ifdef USING_DEVICE_MEMORY
#   undef USING_DEVICE_MEMORY;
#endif


#endif /* SP_DEF_LITE_H_ */
