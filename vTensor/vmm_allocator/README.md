g++ -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` vmm_allocator.cpp -o vmmAllocator`python3-config --extension-suffix` -I python -I/usr/local/cuda/include -L/usr/lib64/ -l cuda -I.

# use
import vmmAllocator
c = vmmAllocator.VmmSegment(10*1024*1024)
c.expandSegment()
