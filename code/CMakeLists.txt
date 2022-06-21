set(CONTROLLER_SRC
#    gen3Dskeleton.cpp
#    touchSens.cpp
)

set(CONTROLLER_HDR
#    ${CMAKE_CURRENT_SOURCE_DIR}/../include/multimodaltouch/gen3Dskeleton.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/../include/multimodaltouch/touchSens.h
)

#=== Library files do not use main() function ===
#add_library(${PROJECT_NAME} SHARED ${CONTROLLER_SRC} ${CONTROLLER_HDR})

#add_library(baxter_gazebo_ros_control_plugin baxter_gazebo_ros_control_plugin.cpp)
#target_link_libraries(baxter_gazebo_ros_control_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} gazebo_ros_control)
#add_dependencies(baxter_gazebo_ros_control_plugin ${catkin_EXPORTED_TARGETS})

#=== Executables
#add_executable(gen3Dskeleton gen3Dskeleton.cpp)
#target_link_libraries(gen3Dskeleton ${catkin_LIBRARIES})
#add_dependencies(gen3Dskeleton multimodaltouch_generate_messages_cpp ${catkin_EXPORTED_TARGETS})

#add_executable(touchSens touchSens.cpp)
#target_link_libraries(touchSens ${catkin_LIBRARIES})
#add_dependencies(touchSens multimodaltouch_generate_messages_cpp ${catkin_EXPORTED_TARGETS})



#add_executable(baxter_mmt_control baxter_mmt_control.cpp)
#target_link_libraries(baxter_mmt_control baxter_gazebo_ros_control_plugin ${catkin_LIBRARIES})

#add_executable(node_name [files])
#target_link_libraries(node_name [custom libraries ... ] {catkin_LIBRARIES} )
