cmake_minimum_required(VERSION 3.16)
project(GPK_CPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)

option(GPK_NATIVE_ARCH "Optimise for the build machine (-march=native)" OFF)
option(GPK_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

function(gpk_target_defaults target)
    target_compile_definitions(${target} PRIVATE _USE_MATH_DEFINES)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(GPK_NATIVE_ARCH)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
        if(GPK_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

find_package(OpenMP REQUIRED)

# ---- Core library ----
add_library(gpk_core STATIC
    src/gpk_thermo.cpp
    src/gpk_gaussian.cpp
    src/gpk_grid.cpp
    src/gpk_geometry.cpp
    src/gpk_physics.cpp
    src/gpk_connectivity.cpp
    src/gpk_transitions.cpp
    src/gpk_transport.cpp
    src/gpk_simulation.cpp
)
target_include_directories(gpk_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_link_libraries(gpk_core PUBLIC OpenMP::OpenMP_CXX)
gpk_target_defaults(gpk_core)

# ---- Unified executable ----
add_executable(gpk demos/gpk.cpp)
target_link_libraries(gpk gpk_core)
gpk_target_defaults(gpk)
target_include_directories(gpk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Statically link the C++/GCC runtimes on MinGW/MSYS2 so the executable does
# not pick up a mismatched libstdc++-6.dll from PATH (e.g. Git's mingw64),
# which otherwise crashes when constructing std::ifstream/std::string.
if(MINGW)
    target_link_options(gpk PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# ---- Benchmark (standalone, no INI) ----
add_executable(benchmark demos/benchmark.cpp)
target_link_libraries(benchmark gpk_core)
gpk_target_defaults(benchmark)
target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Detailed benchmark ----
add_executable(bench_detailed demos/bench_detailed.cpp)
target_link_libraries(bench_detailed gpk_core)
gpk_target_defaults(bench_detailed)
target_include_directories(bench_detailed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Real-materials validation suite ----
add_executable(validate_real_materials demos/validate_real_materials.cpp)
target_link_libraries(validate_real_materials gpk_core)
gpk_target_defaults(validate_real_materials)

# ---- GPK vs MBO/threshold-dynamics benchmark ----
add_executable(bench_gpk_vs_mbo demos/bench_gpk_vs_mbo.cpp)
target_link_libraries(bench_gpk_vs_mbo gpk_core)
gpk_target_defaults(bench_gpk_vs_mbo)

# ---- One-to-one isotropic grain-growth comparison against a calibrated
# multi-order-parameter phase-field reference ----
add_executable(validate_gpk_phase_field demos/validate_gpk_phase_field.cpp)
target_link_libraries(validate_gpk_phase_field gpk_core)
gpk_target_defaults(validate_gpk_phase_field)
target_compile_definitions(validate_gpk_phase_field PRIVATE GPK_BINARY_DIR="${CMAKE_BINARY_DIR}")

# ---- Optional sub-grid micromagnetic extension demo ----
# Separate, opt-in module (include/gpk_micromagnetics.hpp) -- not part of the
# core GPKGrid::step() loop; use when grain-scale coercivity/hysteresis-loop
# statistics are needed beyond the base E_mag(g) driving-force term.
add_executable(micromagnetics_hysteresis demos/micromagnetics_hysteresis.cpp)
target_link_libraries(micromagnetics_hysteresis gpk_core)
gpk_target_defaults(micromagnetics_hysteresis)
target_include_directories(micromagnetics_hysteresis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(micromagnetics_materials_study demos/micromagnetics_materials_study.cpp)
target_link_libraries(micromagnetics_materials_study gpk_core)
gpk_target_defaults(micromagnetics_materials_study)
target_include_directories(micromagnetics_materials_study PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(validate_micromagnetics_v2 demos/validate_micromagnetics_v2.cpp)
target_link_libraries(validate_micromagnetics_v2 gpk_core)
gpk_target_defaults(validate_micromagnetics_v2)

# ---- Extended robustness suites: bicrystal / curvature-crossover /
# grooving-angle / precipitate-shrinkage tests, for the magnetic extensions
# and for the base (non-magnetic) ThermoModel extensions ----
add_executable(validate_magnetic_extended demos/validate_magnetic_extended.cpp)
target_link_libraries(validate_magnetic_extended gpk_core)
gpk_target_defaults(validate_magnetic_extended)
target_include_directories(validate_magnetic_extended PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(validate_extended_base_models demos/validate_extended_base_models.cpp)
target_link_libraries(validate_extended_base_models gpk_core)
gpk_target_defaults(validate_extended_base_models)
target_include_directories(validate_extended_base_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Foundational theory-validation suite: Gauss-Bonnet shape-independent
# area law, von Neumann-Mullins topological law, Mullins linear-stability
# perturbation decay, energy monotonicity, and formal convergence order ----
add_executable(validate_theory_foundations demos/validate_theory_foundations.cpp)
target_link_libraries(validate_theory_foundations gpk_core)
gpk_target_defaults(validate_theory_foundations)
target_include_directories(validate_theory_foundations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(validate_dimensional_multiphase demos/validate_dimensional_multiphase.cpp)
target_link_libraries(validate_dimensional_multiphase gpk_core)
gpk_target_defaults(validate_dimensional_multiphase)
target_include_directories(validate_dimensional_multiphase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Model-independent digital-topology transaction tests ----
add_executable(validate_topology_transactions demos/validate_topology_transactions.cpp)
target_link_libraries(validate_topology_transactions gpk_core)
gpk_target_defaults(validate_topology_transactions)

# ---- Interface-local geometry/storage and thread-determinism regression ----
add_executable(validate_interface_local_geometry demos/validate_interface_local_geometry.cpp)
target_link_libraries(validate_interface_local_geometry gpk_core)
gpk_target_defaults(validate_interface_local_geometry)

# ---- Copy inputs directory next to executables ----
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/inputs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Keep the runtime copies current when an INI file is edited after the initial
# CMake configure step.  Simulations are normally launched from build/ and read
# build/inputs/*.ini, so a configure-time copy alone can otherwise go stale.
add_custom_target(gpk_sync_inputs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs
            ${CMAKE_CURRENT_BINARY_DIR}/inputs
    COMMENT "Synchronising GPK input files")

# ---- Create default output directories ----
foreach(d
    output/grain_growth
    output/grain_growth/isotropic_2d
    output/grain_growth/isotropic_3d
    output/grain_growth/anisotropic_2d
    output/grain_growth/anisotropic_3d
    output/grain_growth/fine_grain_2d
    output/solidification
    output/solidification/dendrite_2d
    output/solidification/dendrite_3d
    output/solidification/equiaxed_2d
    output/solidification/high_undercooling_2d
    output/solidification/stochastic_2d
    output/solidification/elastic_2d
    output/recrystallisation
    output/recrystallisation/standard_2d
    output/recrystallisation/standard_3d
    output/recrystallisation/high_energy_2d
    output/recrystallisation/continuous_nuc_2d
    output/recrystallisation/high_energy_3d
    output/ostwald_ripening
    output/ostwald_ripening/standard_2d
    output/ostwald_ripening/standard_3d
    output/ostwald_ripening/coarse_2d
    output/ostwald_ripening/fine_2d
    output/ostwald_ripening/high_diff_2d
    output/precipitation
    output/precipitation/three_phase_2d
    output/precipitation/three_phase_3d
    output/precipitation/four_phase_2d
    output/precipitation/high_solute_2d
    output/precipitation/low_temperature_2d
    output/wulff_shape
    output/precipitate_breakup
    output/magnetic_annealing
    output/magnetoelastic
    output/ndfeb_wetting
    output/dislocation
    )
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${d})
endforeach()

if(BUILD_TESTING)
    add_test(NAME real_materials COMMAND validate_real_materials)
    add_test(NAME theory_foundations COMMAND validate_theory_foundations)
    add_test(NAME extended_base_models COMMAND validate_extended_base_models)
    add_test(NAME dimensional_multiphase COMMAND validate_dimensional_multiphase)
    add_test(NAME topology_transactions COMMAND validate_topology_transactions)
    add_test(NAME interface_local_geometry COMMAND validate_interface_local_geometry)
    add_test(NAME gpk_phase_field COMMAND validate_gpk_phase_field)
    add_test(NAME magnetic_extended COMMAND validate_magnetic_extended)
    add_test(NAME micromagnetics_v2 COMMAND validate_micromagnetics_v2)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME implementation_regression
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_implementation.py
                         --gpk-bin $<TARGET_FILE:gpk>)
        add_test(NAME literature_regression
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_literature.py
                         --workspace ${CMAKE_CURRENT_SOURCE_DIR}
                         --build-dir ${CMAKE_BINARY_DIR})
        set_tests_properties(implementation_regression literature_regression
                             PROPERTIES ENVIRONMENT "PYTHONUTF8=1")
    endif()
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" GPK_BUILD_TYPE_UPPER)
message(STATUS "CXX flags  : ${CMAKE_CXX_FLAGS_${GPK_BUILD_TYPE_UPPER}}")
message(STATUS "OpenMP     : ${OpenMP_CXX_FLAGS}")
