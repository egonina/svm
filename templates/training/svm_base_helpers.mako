enum KernelType {
  LINEAR,
  POLYNOMIAL,
  GAUSSIAN,
  SIGMOID
};

// ========================== CPU Data Allocation Functions ==========================
// CPU data structures
// Need to be linked to Python
float *data;
float *labels;
float *transposedData;
float *alpha;

float *hostData;
bool hostDataAlloced; // TODO: move this to Python?
bool transposedDataAlloced; // TODO: move this to Python?
int hostPitchInFloats;
size_t devDataPitch;
int sizeOfCache;

float *hostResult;

void alloc_point_data_on_CPU(PyObject *input_data) {
  data = ((float*)PyArray_DATA(input_data));
}

void alloc_labels_on_CPU(PyObject *input_data) {
  labels = ((float*)PyArray_DATA(input_data));
}

void alloc_alphas_on_CPU(PyObject *input_data) {
  alpha = ((float*)PyArray_DATA(input_data));
}

void alloc_host_result_on_CPU() {
    hostResult = (float*)malloc(8*sizeof(float));
}
    
void dealloc_transposed_data_on_CPU() {
    if (transposedDataAlloced) {
      free(transposedData);
    }
}

void dealloc_point_data_on_CPU() {
    if (hostDataAlloced) {
      free(hostData);
    }
    dealloc_transposed_data_on_CPU();
}

void dealloc_host_result_on_CPU() {
    free(hostResult);
}
