FILE(GLOB model_occ_SRC  *.cpp  )
add_library(model_occ ${model_occ_SRC}  )

target_include_directories(model_occ BEFORE PRIVATE ${OCE_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(model_occ    utilities data ${OCE_LIBRARIES} )