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
SET(COPYRIGHT "All rights reserved. (2016 )")


SET(SP_REAL float)

SET(SIMPLA_MAXIMUM_DIMENSION 3)

SET(MAX_NUM_OF_DIMS 10)

#SET(SAMRAI_DIR /pkg/SAMRAI/3.11.0-debug/)

