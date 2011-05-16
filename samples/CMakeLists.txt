
set(NLMAGICK_SOURCE_DIR  "nlmagick_source_dir" CACHE PATH "")
set(NLMAGICK_BUILD_DIR   "nlmagick_build_dir" CACHE PATH "")

add_executable(demo_simple_motion_tracker         demo_simple_motion_tracker.cpp )
target_link_libraries(demo_simple_motion_tracker  ktrack  ${Boost_LIBRARIES}  ${OpenCV_LIBS} )

add_executable(demo_active_contour_tracker         demo_active_contour_tracker.cpp )
target_link_libraries(demo_active_contour_tracker  ktrack  sfm_rbac_lankton ${Boost_LIBRARIES}  ${OpenCV_LIBS} )


