
: ${ALPAKA_CI_HIP_ROOT_DIR?"ALPAKA_CI_HIP_ROOT_DIR must be specified"}
: ${ALPAKA_CI_HIP_BRANCH?"ALPAKA_CI_HIP_BRANCH must be specified"}
: ${CMAKE_BUILD_TYPE?"CMAKE_BUILD_TYPE must be specified"}
: ${CXX?"CXX must be specified"}
: ${CC?"CC must be specified"}
: ${ALPAKA_CI_CMAKE_DIR?"ALPAKA_CI_CMAKE_DIR must be specified"}

# CMake
export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
cmake --version

git clone -b "${ALPAKA_CI_HIP_BRANCH}" --quiet --recursive --single-branch https://github.com/ROCm-Developer-Tools/HIP.git "${ALPAKA_CI_HIP_ROOT_DIR}/source/"
(cd "${ALPAKA_CI_HIP_ROOT_DIR}/source/"; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX=${ALPAKA_CI_HIP_ROOT_DIR} -DBUILD_TESTING=OFF .. && make && make install)
