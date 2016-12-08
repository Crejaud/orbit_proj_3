objects= max_elem.o kernels.o
NVCC= nvcc
opt= -O2 -g -G
ARCH= -arch=sm_30
LIBS=
execname= max_elem

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS)

max_elem.o: max_elem.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c max_elem.cu

clean:
	rm $(objects)
