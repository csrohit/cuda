TARGET	= addition.out



OBJS = gpt.o

CPP_FLAGS = -I/usr/local/cuda-12.1/include



all: ${TARGET}

${TARGET}: ${OBJS}
	nvcc ${CPP_FLAGS} ${OBJS} -o $@


%.o: %.cu
	nvcc ${CPP_FLAGS} -o $@ -c $<

clean:
	rm -rf *.o *.out
