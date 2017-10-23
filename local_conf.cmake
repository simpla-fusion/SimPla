SET(ARCH intel64)
SET(COMPILER $ENV{PRG_ENV})

SET(SIMPLA_VERSION_MAJOR 0)
SET(SIMPLA_VERSION_MINOR 0)
SET(SIMPLA_VERSION_PATCHLEVEL 0)

execute_process(COMMAND git describe --all --dirty --long
        OUTPUT_VARIABLE SIMPLA_VERSION_IDENTIFY
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

SET(AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> ")
SET(COPYRIGHT "All rights reserved. (2017 )")

SET(BUILD_TOOLS ON)

SET(SP_REAL double)

SET(SP_ARRAY_MAX_NDIMS 8)
SET(SP_ARRAY_DEFAULT_ORDER SLOW_FIRST) # SLOW_FIRST ,  FAST_FIRST
SET(SP_ARRAY_INITIALIZE_VALUE SP_SNaN)

SET(SIMPLA_OUTPUT_SUFFIX SimPLA)
#SET(SAMRAI_DIR /pkg/SAMRAI/3.11.0-debug/)

