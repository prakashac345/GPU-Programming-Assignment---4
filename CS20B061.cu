//%%writefile main.cu
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

struct Request
{
    int RequestId;
    int facility;
    int centre;
    int start;
    int slots;
};

//*******************************************

// Write down the kernels here
__global__ void CheckSlot(int index, int startSlot, int endSlot, int *slots, int &flag)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid + startSlot >= endSlot)
        return;
    if (slots[index * 24 + tid + startSlot - 1] - 1 < 0)
    {
        atomicAdd(&flag, 1);
    }
}

__global__ void UpdateSlot(int index, int startSlot, int endSlot, int *slots)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid + startSlot >= endSlot)
        return;
    slots[index * 24 + tid + startSlot - 1]--;
}

__global__ void setSlots(int NF, int *slots, int *capacity)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= NF)
        return;
    tid = tid * 24;
    for (int i = 0; i < 24; i++)
    {
        slots[tid + i] = capacity[tid / 24];
    }
}

__global__ void CountFacilityKernel(int tidCenter, int *slots, int *flags, int *PrefixNoReqPerFac, int *NoReqPerFac, struct Request *sortR, int *succ_reqs)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int index = tidCenter * max_P + tid;

    if (NoReqPerFac[index] == 0)
        return;

    int NoReq;
    if (index >= 1)
        NoReq = PrefixNoReqPerFac[index - 1];
    else
        NoReq = 0;

    int NoReqNext = PrefixNoReqPerFac[index];

    for (int i = NoReq; i < NoReqNext; i++)
    {
        struct Request R = sortR[i];
        int req_cenx = R.centre;
        int req_facx = R.facility;
        int req_startx = R.start;
        int req_slotsx = R.slots;

        if (req_facx == tid && req_cenx == tidCenter)
        {
            flags[index] = 0;

            CheckSlot<<<(req_slotsx + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(index, req_startx, req_startx + req_slotsx, slots, flags[index]);
            cudaDeviceSynchronize();

            if (flags[index] == 0)
            {
                UpdateSlot<<<(req_slotsx + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(index, req_startx, req_startx + req_slotsx, slots);
                cudaDeviceSynchronize();

                atomicAdd(&succ_reqs[tidCenter], 1);
            }
        }
    }
}

__global__ void CountSuccessKernel(int N, int *slots, int *flags, int *PrefixNoReqPerFac, int *NoReqPerFac, int *succ_reqs, int *facility, struct Request *sortR)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N)
        return;

    int NoFac = facility[tid];
    CountFacilityKernel<<<(NoFac + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(tid, slots, flags, PrefixNoReqPerFac, NoReqPerFac, sortR, succ_reqs);
}
__global__ void NoReqPerFacKernel(int R, int *d_NoReqPerFac, struct Request *sortR)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= R)
        return;

    struct Request R1 = sortR[tid];
    int req_cenx = R1.centre;
    int req_facx = R1.facility;
    atomicAdd(&d_NoReqPerFac[req_cenx * max_P + req_facx], 1);
}

//***********************************************

// compare function for qsort to sort requests by centre and facility and then by id
int comp(const void *a, const void *b)
{
    Request *r1 = (Request *)a;
    Request *r2 = (Request *)b;
    if (r1->centre < r2->centre)
        return -1;
    if (r1->centre > r2->centre)
        return 1;
    if (r1->facility < r2->facility)
        return -1;
    if (r1->facility > r2->facility)
        return 1;
    if (r1->RequestId < r2->RequestId)
        return -1;
    if (r1->RequestId > r2->RequestId)
        return 1;
    return 0;
}

int main(int argc, char **argv)
{
    // variable declarations...
    int N, *centre, *facility, *capacity, *fac_ids, *succ_reqs, *tot_reqs;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &N); // N is number of centres

    // Allocate memory on cpu
    centre = (int *)malloc(N * sizeof(int));           // Computer  centre numbers
    facility = (int *)malloc(N * sizeof(int));         // Number of facilities in each computer centre
    fac_ids = (int *)malloc(max_P * N * sizeof(int));  // Facility room numbers of each computer centre
    capacity = (int *)malloc(max_P * N * sizeof(int)); // stores capacities of each facility for every computer centre

    int success = 0;                            // total successful requests
    int fail = 0;                               // total failed requests
    tot_reqs = (int *)malloc(N * sizeof(int));  // total requests for each centre
    succ_reqs = (int *)malloc(N * sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1 = 0, k2 = 0;
    for (int i = 0; i < N; i++)
    {
        fscanf(inputfilepointer, "%d", &centre[i]);
        fscanf(inputfilepointer, "%d", &facility[i]);

        for (int j = 0; j < facility[i]; j++)
        {
            fscanf(inputfilepointer, "%d", &fac_ids[k1]);
            k1++;
        }
        for (int j = 0; j < facility[i]; j++)
        {
            fscanf(inputfilepointer, "%d", &capacity[k2]);
            k2++;
        }
        for (int j = 0; j < max_P - facility[i]; j++)
        {
            fac_ids[k1] = -1;
            capacity[k2] = 0;
            k1++;
            k2++;
        }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots; // Number of slots requested for every request

    // Allocate memory on CPU
    int R;
    fscanf(inputfilepointer, "%d", &R);           // Total requests
    req_id = (int *)malloc((R) * sizeof(int));    // Request ids
    req_cen = (int *)malloc((R) * sizeof(int));   // Requested computer centre
    req_fac = (int *)malloc((R) * sizeof(int));   // Requested facility
    req_start = (int *)malloc((R) * sizeof(int)); // Start slot of every request
    req_slots = (int *)malloc((R) * sizeof(int)); // Number of slots requested for every request

    // Input the user request data
    for (int j = 0; j < R; j++)
    {
        fscanf(inputfilepointer, "%d", &req_id[j]);
        fscanf(inputfilepointer, "%d", &req_cen[j]);
        fscanf(inputfilepointer, "%d", &req_fac[j]);
        fscanf(inputfilepointer, "%d", &req_start[j]);
        fscanf(inputfilepointer, "%d", &req_slots[j]);
        tot_reqs[req_cen[j]] += 1;
    }

    // Sorting the requests by centre and facility and then by id
    struct Request *reqSortR = (struct Request *)malloc((R) * sizeof(struct Request));
    for (int i = 0; i < R; i++)
    {
        reqSortR[i].RequestId = req_id[i];
        reqSortR[i].facility = req_fac[i];
        reqSortR[i].centre = req_cen[i];
        reqSortR[i].start = req_start[i];
        reqSortR[i].slots = req_slots[i];
    }

    qsort(reqSortR, R, sizeof(struct Request), comp);

    int *NoReqPerFac, *d_NoReqPerFac;
    struct Request *d_reqSortR;
    NoReqPerFac = (int *)malloc((max_P * N) * sizeof(int));
    cudaMalloc(&d_NoReqPerFac, (max_P * N) * sizeof(int));
    cudaMalloc(&d_reqSortR, R * sizeof(struct Request));

    cudaMemset(d_NoReqPerFac, 0, (max_P * N) * sizeof(int));
    cudaMemcpy(d_reqSortR, reqSortR, R * sizeof(struct Request), cudaMemcpyHostToDevice);

    // allocate memory on GPU
    int *d_facility, *d_capacity;
    cudaMalloc(&d_facility, N * sizeof(int));
    cudaMalloc(&d_capacity, max_P * N * sizeof(int));
    cudaMemcpy(d_facility, facility, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity, max_P * N * sizeof(int), cudaMemcpyHostToDevice);

    int *succ_reqs_d;
    cudaMalloc(&succ_reqs_d, N * sizeof(int));

    int *d_slots;
    cudaMalloc(&d_slots, 24 * max_P * N * sizeof(int));

    int *d_flags;
    cudaMalloc(&d_flags, max_P * N * sizeof(int));
    cudaMemset(d_flags, 0, max_P * N * sizeof(int));

    //*********************************
    // Call the kernels here

    setSlots<<<(N * max_P - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(N * max_P, d_slots, d_capacity);
    cudaDeviceSynchronize();

    dim3 dimGrid((R - 1) / BLOCKSIZE + 1, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);
    NoReqPerFacKernel<<<dimGrid, dimBlock>>>(R, d_NoReqPerFac, d_reqSortR);
    cudaMemcpy(NoReqPerFac, d_NoReqPerFac, max_P * N * sizeof(int), cudaMemcpyDeviceToHost);

    int *PrefixNoReqPerFac;
    PrefixNoReqPerFac = (int *)malloc((max_P * N) * sizeof(int));
    PrefixNoReqPerFac[0] = NoReqPerFac[0];
    for (int i = 1; i < max_P * N; i++)
    {
        PrefixNoReqPerFac[i] = PrefixNoReqPerFac[i - 1] + NoReqPerFac[i];
    }

    int *d_PrefixNoReqPerFac;
    cudaMalloc(&d_PrefixNoReqPerFac, max_P * N * sizeof(int));
    cudaMemcpy(d_PrefixNoReqPerFac, PrefixNoReqPerFac, max_P * N * sizeof(int), cudaMemcpyHostToDevice);

    CountSuccessKernel<<<(N - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(N, d_slots, d_flags, d_PrefixNoReqPerFac, d_NoReqPerFac, succ_reqs_d, d_facility, d_reqSortR);
    cudaMemcpy(succ_reqs, succ_reqs_d, N * sizeof(int), cudaMemcpyDeviceToHost);

    int total = R;
    for (int i = 0; i < N; i++)
    {
        success += succ_reqs[i];
    }

    fail = total - success;
    //********************************

    // Output
    char *outputfilename = argv[2];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    fprintf(outputfilepointer, "%d %d\n", success, fail);
    for (int j = 0; j < N; j++)
    {
        fprintf(outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j] - succ_reqs[j]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);
    cudaDeviceSynchronize();
    return 0;
}