#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <stdio.h>

#include "pnm/pnm.hpp"

using namespace std;

__constant__ double d_kernel[20];

int kernelSize;
int kernelRadius;

__global__ void grayscale(uint8_t* data, uint8_t* G_data, int channel, size_t data_size)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < data_size)
        {
                uint8_t r, g, b;
                r = data[idx * channel];
                g = data[idx * channel + 1];
                b = data[idx * channel + 2];
                G_data[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
}

__global__ void apply_gaussian_kernel(uint8_t* matrix, uint8_t* result, int kernelSize, int kernelRadius, int numRows, int numCols)
{
        int i = blockIdx.y * blockDim.y + threadIdx.y + kernelRadius;
        int j = blockIdx.x * blockDim.x + threadIdx.x + kernelRadius;

        if (i < numRows - kernelRadius && j < numCols - kernelRadius)
        {
                double sum = 0.0;
                for (int k = 0; k < kernelSize; ++k)
                {
                        for (int l = 0; l < kernelSize; ++l)
                        {
                                sum += d_kernel[k * kernelSize + l] * matrix[(i - kernelRadius + k) * numCols + (j - kernelRadius + l)];
                        }
                }
                result[i * numCols + j] = static_cast<uint8_t>(sum);
        }
}

vector<int> print(vector<uint8_t> &data, ifstream &ifs)
{
    vector<int> size;
    size.resize(3);
    PNM::Info info;
    ifs >> PNM::load( data, info );
    if( true == info.valid() )
    {
        cout << "width = " << info.width() << endl;
        cout << "height = "  << info.height() << endl;
        cout << "maximum = "  << info.maximum() << endl;
        cout << "channel = "  << info.channel() << endl;
        cout << "type = "  << (int)info.type() << endl;
        size[0] = info.width();
        size[1] = info.height();
        size[2] = info.channel();
    }
    else
    {
        cout << "Error type" << endl;
    }
    return size;
}

vector<vector<uint8_t>> vectorToMatrix(vector<uint8_t> data, int width, int height)
{
    vector<vector<uint8_t>> matrix;
    int dataCounter = 0;
    matrix.resize(height);
    for(int i = 0; i < height; i++)
    {
        matrix[i].resize(width);
        for(int j = 0; j < width; j++)
        {
            matrix[i][j] = data[dataCounter];
            dataCounter++;
        }
    }
    return matrix;
}

vector<uint8_t> matrixToVector(vector<vector<uint8_t>>& matrix)
{
    vector<uint8_t> data;
    for (const auto& row : matrix)
    {
        for (uint8_t val : row)
        {
            data.push_back(val);
        }
    }
    return data;
}

vector<double> matrixToVector(vector<vector<double>>& matrix)
{
    vector<double> data;
    for (const auto& row : matrix)
    {
        for (double val : row)
        {
            data.push_back(val);
        }
    }
    return data;
}

vector<uint8_t> convertToVector(uint8_t* data_ptr, size_t data_size)
{
    vector<uint8_t> data_vector(data_ptr, data_ptr + data_size);
    return data_vector;
}

void grayscale(vector<uint8_t> &G_data, vector<uint8_t> data, int channel)
{
        uint8_t* G_data_kernel;
        uint8_t* data_kernel;
        cudaError_t err = cudaSuccess;

        err = cudaMalloc((void**)&data_kernel, data.size());

        err = cudaMemcpy(data_kernel, data.data(), data.size(), cudaMemcpyHostToDevice);

        err = cudaMalloc((void**)&G_data_kernel, G_data.size());

        err = cudaMemcpy(G_data_kernel, G_data.data(), data.size(), cudaMemcpyHostToDevice);

        int threadsPerBlock = 1024;
        int blocksPerGrid = (G_data.size() + threadsPerBlock - 1) / threadsPerBlock;

        grayscale<<<blocksPerGrid, threadsPerBlock>>>(data_kernel, G_data_kernel, channel, G_data.size());

        err = cudaMemcpy(G_data.data(), G_data_kernel, G_data.size(), cudaMemcpyDeviceToHost);
}

void kernel_const_memory(vector<vector<double>>& kernel)
{
    vector<double> kernel_v = matrixToVector(kernel);
    cudaError_t err = cudaMemcpyToSymbol(d_kernel, kernel_v.data(), kernelSize * kernelSize * sizeof(double), 0, cudaMemcpyHostToDevice);
    cout << "Error type: " << cudaGetErrorString(err) << endl;
}

void apply_gaussian_kernel(vector<vector<uint8_t>>& matrix)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numRows = matrix.size();
    int numCols = matrix[0].size();

    vector<uint8_t> matrix_v = matrixToVector(matrix);

    uint8_t* d_matrix;
    uint8_t* d_result;
    cudaMalloc((void**)&d_matrix, numRows * numCols * sizeof(uint8_t));
    cudaMalloc((void**)&d_result, numRows * numCols * sizeof(uint8_t));

    cudaMemcpy(d_matrix, matrix_v.data(), numRows * numCols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    apply_gaussian_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, kernelSize, kernelRadius, numRows, numCols);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << endl << "Time spent on operations (ordinary):" << milliseconds << endl;

    uint8_t* test_v = (uint8_t*)malloc(numRows * numCols * sizeof(uint8_t));

    cudaMemcpy(matrix_v.data(), d_result, numRows * numCols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    matrix = vectorToMatrix(matrix_v, matrix.size(), matrix[0].size());

    cudaFree(d_matrix);
    cudaFree(d_result);
}

void apply_gaussian_kernel_s_m(vector<vector<uint8_t>>& matrix)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int numRows = matrix.size();
    int numCols = matrix[0].size();

    vector<uint8_t> matrix_v = matrixToVector(matrix);

    uint8_t* d_matrix;
    uint8_t* d_result;

    uint8_t* d_matrix_f = nullptr;
    uint8_t* d_result_f = nullptr;

    cudaMallocHost((uint8_t **)&d_matrix_f, matrix_v.size() * sizeof(uint8_t));
    cudaMallocHost((uint8_t **)&d_result_f, matrix_v.size() * sizeof(uint8_t));
    cudaMalloc((void**)&d_matrix, numRows * numCols * sizeof(uint8_t));
    cudaMalloc((void**)&d_result, numRows * numCols * sizeof(uint8_t));

    memcpy(d_matrix_f, matrix_v.data(), numRows * numCols * sizeof(uint8_t));

    cudaMemcpy(d_matrix, d_matrix_f, numRows * numCols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    apply_gaussian_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, kernelSize, kernelRadius, numRows, numCols);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << endl << "Time spent on operations (fixed):" << milliseconds << endl;

    cudaMemcpy(matrix_v.data(), d_result, numRows * numCols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    matrix = vectorToMatrix(matrix_v, matrix.size(), matrix[0].size());

    cudaFreeHost(d_matrix_f);
    cudaFreeHost(d_result_f);
    cudaFree(d_matrix);
    cudaFree(d_result);
}

void apply_gaussian_kernel_v_m(vector<vector<uint8_t>>& matrix)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int numRows = matrix.size();
    int numCols = matrix[0].size();

    vector<uint8_t> matrix_v = matrixToVector(matrix);

    uint8_t* d_matrix_f = nullptr;
    uint8_t* d_result_f = nullptr;

    cudaHostAlloc((uint8_t**)&d_matrix_f, numRows * numCols * sizeof(uint8_t), cudaHostAllocMapped);
    cudaHostAlloc((uint8_t**)&d_result_f, numRows * numCols * sizeof(uint8_t), cudaHostAllocMapped);

    memcpy(d_matrix_f, matrix_v.data(), numRows * numCols * sizeof(uint8_t));

    uint8_t* d_matrix = nullptr;
    uint8_t* d_result = nullptr;

    cudaHostGetDevicePointer((void**)&d_matrix, d_matrix_f, 0);
    cudaHostGetDevicePointer((void**)&d_result, d_result_f, 0);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    apply_gaussian_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, kernelSize, kernelRadius, numRows, numCols);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << endl << "Time spent on operations (showed):" << milliseconds << endl;

    cudaMemcpy(matrix_v.data(), d_result_f, numRows * numCols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    matrix = vectorToMatrix(matrix_v, matrix.size(), matrix[0].size());

    cudaFreeHost(d_matrix_f);
    cudaFreeHost(d_result_f);
}

void apply_gaussian_kernel_u_m(vector<vector<uint8_t>>& matrix)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numRows = matrix.size();
    int numCols = matrix[0].size();

    vector<uint8_t> matrix_v = matrixToVector(matrix);

    uint8_t* d_matrix;
    uint8_t* d_result;
    cudaMallocManaged((void**)&d_matrix, numRows * numCols * sizeof(uint8_t), cudaMemAttachGlobal);
    cudaMallocManaged((void**)&d_result, numRows * numCols * sizeof(uint8_t), cudaMemAttachGlobal);

    cudaMemcpy(d_matrix, matrix_v.data(), numRows * numCols * sizeof(uint8_t), cudaMemcpyHostToHost);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    apply_gaussian_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, kernelSize, kernelRadius, numRows, numCols);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << endl << "Time spent on operations (ordinary):" << milliseconds << endl;

    uint8_t* test_v = (uint8_t*)malloc(numRows * numCols * sizeof(uint8_t));

    cudaMemcpy(matrix_v.data(), d_result, numRows * numCols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    matrix = vectorToMatrix(matrix_v, matrix.size(), matrix[0].size());

    cudaFree(d_matrix);
    cudaFree(d_result);
}

void applyGaussianKernel(vector<vector<uint8_t>>& matrix, vector<vector<double>>& kernel)
{
    int kernelSize = kernel.size();
    int kernelRadius = kernelSize / 2;
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    vector<vector<uint8_t>> result(numRows, vector<uint8_t>(numCols, 0.0));

    for (int i = kernelRadius; i < numRows - kernelRadius; ++i)
    {
        for (int j = kernelRadius; j < numCols - kernelRadius; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < kernelSize; ++k)
            {
                for (int l = 0; l < kernelSize; ++l)
                {
                    sum += kernel[k][l] * matrix[i - kernelRadius + k][j - kernelRadius + l];
                }
            }
            result[i][j] = static_cast<uint8_t>(sum);
        }
    }

    matrix = result;
}

vector<vector<double>> gaussianMatrix(double sigma, int matrixSize)
{
    vector<vector<double>> kernel(matrixSize, vector<double>(matrixSize, 0.0));
    double sum = 0;
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            int x = i - matrixSize / 2;
            int y = j - matrixSize / 2;
            kernel[i][j] = 1 / (2 * M_PI * pow(sigma, 2)) * exp(-1 * ((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))));
            sum += kernel[i][j];
        }
    }

    cout << "Summary: " << sum << endl;
    double sum2 = 0;
    cout << "Gaussian Kernel Matrix:" << endl;
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            cout << kernel[i][j] << " ";
            kernel[i][j] /= sum;
            sum2 += kernel[i][j];
        }
        cout << endl;
    }

    cout << "Second summary: " << sum2 << endl;

    return kernel;
}

int main( int argc, char *argv[] )
{
    ifstream ifs( "./" "/Cat.ppm", std::ios_base::binary );
    vector<uint8_t> data;
    vector<int> size;
    int width, height, channel;
    uint8_t r, g, b;

    size = print(data, ifs);
    width = size[0];
    height = size[1];
    channel = size[2];

    vector<uint8_t> G_data;
    G_data.resize(width * height);

    grayscale(G_data, data, channel);

    double sigma = 1.0;
    int matrixSize = 3;

    vector<vector<double>> matrix = gaussianMatrix(sigma, matrixSize);

    vector<vector<uint8_t>> image = vectorToMatrix(G_data, width, height);

    kernelSize = matrix.size();
    kernelRadius = matrix.size() / 2;

    kernel_const_memory(matrix);

    // apply_gaussian_kernel(image);

    // apply_gaussian_kernel_s_m(image);

    // apply_gaussian_kernel_v_m(image);

    apply_gaussian_kernel_u_m(image);

    G_data = matrixToVector(image);

    cout << "size new vector: " << data.size() << endl;

    cout << "GDATA: " << G_data.size() << endl;

    PNM::Info save;
    save.type() = PNM::P2;
    std::ofstream ofs("Cat.pgm", std::ios::binary);

    if (!ofs)
    {
        cerr << "Error opening output file" << endl;
        return 1;
    }

    ofs << "P2\n";
    ofs << width << " " << height << "\n";
    ofs << "255\n";

    for (int i = 0; i < G_data.size(); ++i)
    {
        ofs << static_cast<int>(G_data[i]) << " ";
        if ((i + 1) % width == 0)
        {
            ofs << "\n";
        }
    }

    return 0;
}
