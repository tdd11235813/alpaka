## Compiling HIP from Source

Follow [this](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md "HIP installation") guide for installing HIP.
HIP requires either `nvcc` or `hcc` to be installed on your system (see guide for further details).

- If you want the hip binaries to be located in a directory that does not require superuser access, be sure to change the install directory of HIP by modifying the `CMAKE_INSTALL_PREFIX` cmake variable.
- Also, after the installation is complete, add the following line to the .profile file in your home directory, in order to add the path to the HIP binaries to PATH:
`PATH=$PATH:<path_to_binaries>`

```bash
git clone --recursive https://github.com/ROCm-Developer-Tools/HIP.git
cd "HIP"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX=${YOUR_HIP_INSTALL_DIR} -DBUILD_TESTING=OFF ..
make
make install
```
Set the appropriate paths (edit `${YOUR_**}` variables).
```bash
# HIP_PATH required by HIP tools
export HIP_PATH=${YOUR_HIP_INSTALL_DIR}
# Paths required by HIP tools
export CUDA_PATH=${YOUR_CUDA_ROOT}
# - if required, path to HCC compiler. Default /opt/rocm/hcc.
export HCC_HOME=${YOUR_HCC_ROOT}
# - if required, path to HSA include, lib. Default /opt/rocm/hsa.
export HSA_PATH=${YOUR_HSA_PATH}
# HIP binaries and libraries
export PATH=${HIP_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${HIP_PATH}/lib64:${LD_LIBRARY_PATH}
```
Test the HIP binaries.
```bash
# calls nvcc or hcc
which hipcc
hipcc -V
which hipconfig
hipconfig -v
```


## Verifying HIP installation
- If PATH points to the location of the HIP binaries, the following command should list several relevant environment variables, and also the selected compiler on your system-`hipconfig -f`
- Compile and run the [square sample](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square), as pointed out in the [original](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#verify-your-installation) HIP install guide.

## Compiling examples with HIP back-end
As of now, the back-end has only been tested on the NVIDIA platform.
### NVIDIA Platform
* One issue in this branch of alpaka is that the host compiler flags don't propagate to the device compiler, as they do in CUDA. This is because a counterpart to the CUDA_PROPAGATE_HOST_FLAGS cmake variable has not been defined in the FindHIP.cmake file.
Alpaka forwards the host compiler flags in cmake to the `HIP_NVCC_FLAGS` cmake variable, which also takes user-given flags. To add flags to this variable, toggle the advanced mode in `ccmake`.


## Random Number Generator Library rocRAND for HIP back-end

rocRAND provides an interface for HIP, where the cuRAND or rocRAND API is called depending on the chosen HIP platform (can be configured with cmake in alpaka).

Clone the rocRAND repository, then build and install it:
```bash
git clone https://github.com/ROCmSoftwarePlatform/rocRAND
cd rocRAND
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${HIP_PATH} -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF -DCMAKE_MODULE_PATH=${HIP_PATH}/cmake ..
make
```

The `CMAKE_MODULE_PATH` is a cmake variable for locating module finding scripts like *FindHIP.cmake*.
The paths to the `rocRAND` library and include directories should be appended to the `CMAKE_PREFIX_PATH` variable.
