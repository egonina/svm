#include <math.h>
#include <float.h>

// =================== INITIALIZE ===================
void launchInitialization(float* devData,
                          int devDataPitchInFloats,
                          int nPoints, int nDimension, int kType,
                          float parameterA, float parameterB, float parameterC,
                          float* devKernelDiag, float* devAlpha, float* devF,
                          float* devLabels, dim3 blockConfig, dim3 threadConfig);

void launchTakeFirstStep(void* devResult, float* devKernelDiag,
                         float* devData, int devDataPitchInFloats,
                         float* devAlpha, float cost, int nDimension,
                         int iLow, int iHigh, int kType, 
                         float parameterA, float parameterB, float parameterC,
                         dim3 blockConfig, dim3 threadConfig); 

// =============== FIRSTORDER =============
int firstOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension);

int firstOrderPhaseTwoSize();

void launchFirstOrder(bool iLowCompute, bool iHighCompute,
                      int kType, int nPoints, int nDimension,
                      dim3 blocksConfig, dim3 threadsConfig,
                      dim3 globalThreadsConfig, float* devData,
                      int devDataPitchInFloats, float* devTransposedData,
                      int devTransposedDataPitchInFloats, float* devLabels,
                      float epsilon, float cEpsilon, float* devAlpha,
                      float* devF, float sAlpha1Diff, float sAlpha2Diff,
                      int iLow, int iHigh, float parameterA, float parameterB,
                      float parameterC, float* devCache, int devCachePitchInFloats,
                      int iLowCacheIndex, int iHighCacheIndex, int* devLocalIndicesRL,
                      int* devLocalIndicesRH, float* devLocalFsRH, float* devLocalFsRL,
                      float* devKernelDiag, void* devResult, float cost);

// ================= SECONDORDER =================
int secondOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension);

int secondOrderPhaseTwoSize();

int secondOrderPhaseThreeSize(bool iHighCompute, int nDimension);

int secondOrderPhaseFourSize();

void launchSecondOrder(bool iLowCompute, bool iHighCompute,
                       int kType, int nPoints, int nDimension, dim3 blocksConfig,
                       dim3 threadsConfig, dim3 globalThreadsConfig, float* devData,
                       int devDataPitchInFloats, float* devTransposedData,
                       int devTransposedDataPitchInFloats, float* devLabels, float epsilon,
                       float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff,
                       float sAlpha2Diff, int iLow, int iHigh, float parameterA,
                       float parameterB, float parameterC, Cache* kernelCache,
                       float* devCache, int devCachePitchInFloats, int iLowCacheIndex,
                       int iHighCacheIndex, int* devLocalIndicesRH, float* devLocalFsRH,
                       float* devLocalFsRL, int* devLocalIndicesMaxObj,
                       float* devLocalObjsMaxObj, float* devKernelDiag,
                       void* devResult, float* hostResult, float cost, int iteration);
