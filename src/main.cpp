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

#define MAXDIGITS 5

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

int determineThreadCount(const int processorCount, const int matrixARows)
{
  if (matrixARows <= processorCount)
  {
    return matrixARows;
  }
  return processorCount;
}

int getMaxAmountOfDigits(const int rows, const int columns, int **arr)
{
  int max = 0;
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < columns; ++c)
    {
      int num = arr[r][c];
      int digits = log10(num);
      max = std::max(digits, max);
    }
  }
  return max;
}

void printMatrix(const int rows, const int columns, int **arr)
{
  const int maxDigits = getMaxAmountOfDigits(rows, columns, arr);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < columns; ++c)
    {
      int num = arr[r][c];
      int digits = log10(num);
      std::cout << num;
      for (; digits <= maxDigits; ++digits)
      {
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

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

void calculateResultMatrix(int **matrixA, const int aRows, const int aCols, int **matrixB, const int bCols)
{
  int **resultMatrix = getMatrix(aRows, bCols, false);

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

  // std::cout << "Matrix AB: \n";
  // printMatrix(aRows, bCols, resultMatrix);

  for (int h = 0; h < aRows; h++)
  {
    delete[] resultMatrix[h];
  }
  delete[] resultMatrix;
}

void calculateResultMatrixParallel(int **matrixA, const int aRows, const int aCols, int **matrixB, const int bCols)
{

  int **resultMatrix = getMatrix(aRows, bCols, false);

  const auto processor_count = omp_get_num_procs();
  const int thread_count =
      determineThreadCount(processor_count, aRows);
  const std::vector<std::pair<int, int>> rowStartRowEnd =
      determineRowDistribution(thread_count, aRows);

  omp_set_num_threads(thread_count + 1); // +1 is necessary to get desired amount

  std::cout << "cores: " << processor_count << " threads used: " << thread_count << "\n";

#pragma omp parallel 
  {
    const int id = omp_get_thread_num();
    if (id)
    {
      const int startRow = rowStartRowEnd.at(id - 1).first; // id starts with 1
      const int endRow = rowStartRowEnd.at(id - 1).second;

      for (int currentRow = startRow; currentRow <= endRow; ++currentRow)
      {
        for (int c2 = 0; c2 < bCols; ++c2)
        {
          int result = 0;
          for (int r2 = 0; r2 < aCols; ++r2)
          {
            result += matrixA[currentRow][r2] * matrixBCopy[r2][c2];
          }
          resultMatrix[currentRow][c2] = result;
        }
      }
    }
  }

  // std::cout << "Matrix AB: \n";
  // printMatrix(aRows, bCols, resultMatrix);

  for (int h = 0; h < aRows; h++)
  {
    delete[] resultMatrix[h];
  }
  delete[] resultMatrix;
}

int main(void)
{
  int aRows = 0;
  int aCols = 0;
  int bRows = 0;
  int bCols = 0;

  std::cout << "Enter the number of rows for Matrix A " << std::endl;
  std::cin >> aRows;

  std::cout << "Enter the number of columns Matrix A:  "
            << std::endl;
  std::cin >> aCols;

  std::cout << "Enter the number of columns Matrix B: "
            << std::endl;
  std::cin >> bRows;

  bCols = aCols;

  int **matrixA = getMatrix(aRows, aCols);
  int **matrixB = getMatrix(bCols, bRows);

  std::chrono::steady_clock::time_point beginSequential = std::chrono::steady_clock::now();
  calculateResultMatrix(matrixA, aRows, aCols, matrixB, bCols);
  std::chrono::steady_clock::time_point endSequential = std::chrono::steady_clock::now();
  auto linearTime = std::chrono::duration_cast<std::chrono::microseconds>(endSequential - beginSequential).count();
  std::cout << "sequential version: " << linearTime << "[µs]" << std::endl;

  std::chrono::steady_clock::time_point beginParallel = std::chrono::steady_clock::now();
  calculateResultMatrixParallel(matrixA, aRows, aCols, matrixB, bCols);
  std::chrono::steady_clock::time_point endParallel = std::chrono::steady_clock::now();
  auto parallelTime = std::chrono::duration_cast<std::chrono::microseconds>(endParallel - beginParallel).count();
  std::cout << "parallel version: " << parallelTime << "[µs]" << std::endl;

  // std::cout << "Matrix A: \n";
  // printMatrix(aRows, aCols, matrixA);
  // std::cout << "Matrix B: \n";
  // printMatrix(bCols, bRows, matrixB);

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

  return 0;
}