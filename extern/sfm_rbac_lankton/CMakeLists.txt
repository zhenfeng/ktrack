file( GLOB  SFM_LOCAL_CHANVESE_SRC  include/*.txx include/*.h  src/*.cpp)
add_library( sfm_rbac_lankton   ${SFM_LOCAL_CHANVESE_SRC}  )
target_link_libraries( sfm_rbac_lankton ${OpenCV_LIBS} )
