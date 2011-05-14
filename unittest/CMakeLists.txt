# Define executable: test simple tracking on webcam
add_executable(test_webcam_pfcore         test_webcam_pfcore.cpp )
target_link_libraries(test_webcam_pfcore  ktrack  ${Boost_LIBRARIES}  ${OpenCV_LIBS} )


# Define executable: webcam test of matching to get frame-to-frame motion
# TODO: this is a sample, connects to nlmagick
#add_executable(webcam_frame2frame  test_opencv_frame2frame.cpp )
#target_link_libraries(webcam_frame2frame  ktrack  ${Boost_LIBRARIES} ${OpenCV_LIBS}  )


# Define executable: click, drag, track!
# TODO: this is a *sample*
#add_executable(webcam_click_drag_track  test_click_drag_track.cpp )
#target_link_libraries(webcam_click_drag_track  ktrack  ${Boost_LIBRARIES} ${OpenCV_LIBS}  )


