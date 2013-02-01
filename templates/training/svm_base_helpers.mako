#define intDivideRoundUp(a, b) (a%b!=0)?(a/b+1):(a/b)

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
float *hostResult;

float *hostData;
bool hostDataAlloced;
bool transposedDataAlloced;
int hostPitchInFloats;
size_t devDataPitch;
int sizeOfCache;

void alloc_point_data_on_CPU(PyObject *input_data) {
  data = ((float*)PyArray_DATA(input_data));
}

void alloc_labels_on_CPU(PyObject *input_labels) {
  labels = ((float*)PyArray_DATA(input_labels));
}

void alloc_alphas_on_CPU(PyObject *input_alphas) {
  alpha = ((float*)PyArray_DATA(input_alphas));
}

void alloc_result_on_CPU(PyObject *input_result) {
  hostResult = ((float*)PyArray_DATA(input_result));
}
