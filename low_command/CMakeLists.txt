cmake_minimum_required(VERSION 3.0.2)
project(low_command)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
  unitree_legged_msgs

  rospy
)

catkin_package(
	CATKIN_DEPENDS
)

message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
else()
  set(ARCH arm64)
endif()

link_directories(${CMAKE_SOURCE_DIR}/00000000/lib/cpp/${ARCH})

set(EXTRA_LIBS -pthread libunitree_legged_sdk.so)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/unitree_legged_sdk/include
)



#add_executable(low_communication
#	src/low_communication.cpp
#)
# target_link_libraries(
# 	low_communication
# 	${catkin_LIBRARIES}
# )

catkin_install_python(PROGRAMS src/low_communication.py
  	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# install(TARGETS	low_communication
#   	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)