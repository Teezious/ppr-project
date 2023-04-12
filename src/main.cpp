#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <signal.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#define MAXDIGITS 8 // define the max amount of digits a number inside Matrix A or B can have
#define MINIMUMCOMPUTATIONSPERTHREAD 20000000 // estimate to increase the bad performance of small matrices by not using the max amount of threads
#define LOOPRUNS 10 // defines the amount of benchmark runs


int **getMatrix(int rows, int columns, bool random = true)
{
  int **array2D = 0;
  const int maxNum = pow(10, MAXDIGITS - 1);
  array2D = new int *[rows];

  for (int r = 0; r < rows; r++)
  {
    array2D[r] = new int[columns];
    if (random)
    {
      for (int c = 0; c < columns; c++)
      {
        array2D[r][c] = rand() % maxNum;
      }
    }
    else
    {
      for (int c = 0; c < columns; c++)
      {
        array2D[r][c] = 0;
      }
    }
  }
  return array2D;
}

// determines the amount of threads to be used
int determineThreadCount(const int processorCount, const int matrixARows, const int matrixACols, const int matrixBCols)
{
  const unsigned long long int computationsPerColumn = matrixACols * matrixACols; // multiplication and summation
  const unsigned long long int totalComputations = matrixARows * computationsPerColumn * matrixBCols;
  const int threads = std::max(1, static_cast<int>(totalComputations / MINIMUMCOMPUTATIONSPERTHREAD));
  return (threads <= processorCount) ? threads : processorCount;
}

// splits the rows of Matrix A in equal sized parts
// each pair represents starting- and end row
std::vector<std::pair<int, int>> determineRowDistribution(const int threads,
                                                          const int tasks)
{
  std::vector<std::pair<int, int>> rowStartRowEnd;
  int index = 0;
  for (int i = 0; i < threads; ++i)
  {
    int tasks_for_this_thread = tasks / threads + (i < tasks % threads);
    rowStartRowEnd.push_back(
        std::make_pair(index, index + tasks_for_this_thread - 1));
    index += tasks_for_this_thread;
  }
  return rowStartRowEnd;
}

template<typename T> auto calculateResultMatrix(int **matrixA, const int aRows, const int aCols, int **matrixB, const int bCols)
{
  std::chrono::steady_clock::time_point beginSequential = std::chrono::steady_clock::now();
  int **resultMatrix = getMatrix(aRows, bCols, false);

  // Matrix multiplication 
  for (int row = 0; row < aRows; ++row)
  {
    for (int c2 = 0; c2 < bCols; ++c2)
    {
      int result = 0;
      for (int r2 = 0; r2 < aCols; ++r2)
      {
        result += matrixA[row][r2] * matrixB[r2][c2];
      }
      resultMatrix[row][c2] = result;
    }
  }

  for (int h = 0; h < aRows; h++)
  {
    delete[] resultMatrix[h];
  }
  delete[] resultMatrix;
  std::chrono::steady_clock::time_point endSequential = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<T>(endSequential - beginSequential);
}

template<typename T> auto calculateResultMatrixParallel(int **matrixA, const int aRows, const int aCols, int **matrixB, const int bCols)
{
  std::chrono::steady_clock::time_point beginParallel = std::chrono::steady_clock::now();

  int **resultMatrix = getMatrix(aRows, bCols, false);

  const auto processor_count = omp_get_num_procs();
  const int thread_count =
      determineThreadCount(processor_count - 1, aRows, aCols, bCols);
  const std::vector<std::pair<int, int>> rowStartRowEnd =
      determineRowDistribution(thread_count, aRows);

  omp_set_num_threads(thread_count);

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    if (id)
    {
      const int startRow = rowStartRowEnd.at(id - 1).first; // id starts with 1
      const int endRow = rowStartRowEnd.at(id - 1).second;

      // Matrix Multiplication for the section of start row to end row
      for (int currentRow = startRow; currentRow <= endRow; ++currentRow)
      {
        for (int c2 = 0; c2 < bCols; ++c2)
        {
          int result = 0;
          for (int r2 = 0; r2 < aCols; ++r2)
          {
            result += matrixA[currentRow][r2] * matrixB[r2][c2];
          }
          resultMatrix[currentRow][c2] = result;
        }
      }
    }
  }

  for (int h = 0; h < aRows; h++)
  {
    delete[] resultMatrix[h];
  }
  delete[] resultMatrix;

  std::chrono::steady_clock::time_point endParallel = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<T>(endParallel - beginParallel);
}

void benchmarkMatrix(const int aRows, const int aCols, const int bRows, const int loopRuns)
{
  const int bCols = aCols;

  std::chrono::milliseconds totalLinearTime = std::chrono::milliseconds(0);
  std::chrono::milliseconds totalParallelTime = std::chrono::milliseconds(0);
  for (int i = 0; i < loopRuns; ++i)
  {
    int **matrixA = getMatrix(aRows, aCols); // using a 1d array, vectors, or boost multi_arrays might be better here
    int **matrixB = getMatrix(bCols, bRows);

    auto linearTime = calculateResultMatrix<std::chrono::microseconds>(matrixA, aRows, aCols, matrixB, bCols);
    totalLinearTime += std::chrono::duration_cast<std::chrono::milliseconds>(totalLinearTime + linearTime);
    auto parallelTime = calculateResultMatrixParallel<std::chrono::microseconds>(matrixA, aRows, aCols, matrixB, bCols);
    totalParallelTime += std::chrono::duration_cast<std::chrono::milliseconds>(totalParallelTime + parallelTime);
    for (int h = 0; h < aRows; h++)
    {
      delete[] matrixA[h];
    }
    delete[] matrixA;
    for (int h = 0; h < bCols; h++)
    {
      delete[] matrixB[h];
    }
    delete[] matrixB;
  }

  std::cout << "\n----------------------------------------------------------------------------\n";
  std::cout << "Dimensions: " << aRows << " x " << aCols << " * " << bRows << " x " << bCols << std::endl;
  std::cout << "Average Linear Time: " << totalLinearTime.count() / loopRuns << "ms\n";
  std::cout << "Average Parallel Time: " << totalParallelTime.count() / loopRuns << "ms\n";
  std::cout << "----------------------------------------------------------------------------";

}

int main(void)
{
  benchmarkMatrix(10,10,10,LOOPRUNS);
  benchmarkMatrix(100,10,10,LOOPRUNS);
  benchmarkMatrix(100,100,10,LOOPRUNS);
  benchmarkMatrix(100,100,100,LOOPRUNS);
  benchmarkMatrix(1000,100,100,LOOPRUNS);
  benchmarkMatrix(1000,1000,100,LOOPRUNS);
  benchmarkMatrix(1000,1000,1000,LOOPRUNS);

  // calculation time gets rather long here 

  // benchmarkMatrix(10000,1000,1000,LOOPRUNS);
  // benchmarkMatrix(10000,10000,1000,LOOPRUNS);
  // benchmarkMatrix(10000,10000,10000,LOOPRUNS);
  return 0;
}