NVCC := nvcc
TARGET := knn_cuda

SRCS := main.cu knn.cu

NVCC_FLAGS := -O3 -arch=sm_70

LIBS := 

INCLUDES := -I. 

all:$(TARGET)


$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $@ $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
