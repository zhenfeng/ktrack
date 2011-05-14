# Define executable: test simple tracking on webcam
add_executable(webcam_track   test_opencv_tracker.cpp )
target_link_libraries(webcam_track  ktrack  ${Boost_LIBRARIES}  ${OpenCV_LIBS} )


# Define executable: webcam test of matching to get frame-to-frame motion
add_executable(webcam_frame2frame  test_opencv_frame2frame.cpp )
target_link_libraries(webcam_frame2frame  ktrack  ${Boost_LIBRARIES} ${OpenCV_LIBS}  )


# Define executable: click, drag, track!
add_executable(webcam_click_drag_track  test_click_drag_track.cpp )
target_link_libraries(webcam_click_drag_track  ktrack  ${Boost_LIBRARIES} ${OpenCV_LIBS}  )


