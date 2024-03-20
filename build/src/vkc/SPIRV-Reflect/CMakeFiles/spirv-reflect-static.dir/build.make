# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/void/Code/cpp/octree_sdf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/void/Code/cpp/octree_sdf/build

# Include any dependencies generated for this target.
include src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/compiler_depend.make

# Include the progress variables for this target.
include src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/progress.make

# Include the compile flags for this target's objects.
include src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/flags.make

src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o: src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/flags.make
src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o: /home/void/Code/cpp/octree_sdf/src/vkc/SPIRV-Reflect/spirv_reflect.c
src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o: src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/void/Code/cpp/octree_sdf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o"
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o -MF CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o.d -o CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o -c /home/void/Code/cpp/octree_sdf/src/vkc/SPIRV-Reflect/spirv_reflect.c

src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.i"
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/void/Code/cpp/octree_sdf/src/vkc/SPIRV-Reflect/spirv_reflect.c > CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.i

src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.s"
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/void/Code/cpp/octree_sdf/src/vkc/SPIRV-Reflect/spirv_reflect.c -o CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.s

# Object files for target spirv-reflect-static
spirv__reflect__static_OBJECTS = \
"CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o"

# External object files for target spirv-reflect-static
spirv__reflect__static_EXTERNAL_OBJECTS =

src/vkc/SPIRV-Reflect/libspirv-reflect-static.a: src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/spirv_reflect.c.o
src/vkc/SPIRV-Reflect/libspirv-reflect-static.a: src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/build.make
src/vkc/SPIRV-Reflect/libspirv-reflect-static.a: src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/void/Code/cpp/octree_sdf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libspirv-reflect-static.a"
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && $(CMAKE_COMMAND) -P CMakeFiles/spirv-reflect-static.dir/cmake_clean_target.cmake
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spirv-reflect-static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/build: src/vkc/SPIRV-Reflect/libspirv-reflect-static.a
.PHONY : src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/build

src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/clean:
	cd /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect && $(CMAKE_COMMAND) -P CMakeFiles/spirv-reflect-static.dir/cmake_clean.cmake
.PHONY : src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/clean

src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/depend:
	cd /home/void/Code/cpp/octree_sdf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/void/Code/cpp/octree_sdf /home/void/Code/cpp/octree_sdf/src/vkc/SPIRV-Reflect /home/void/Code/cpp/octree_sdf/build /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect /home/void/Code/cpp/octree_sdf/build/src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/vkc/SPIRV-Reflect/CMakeFiles/spirv-reflect-static.dir/depend

