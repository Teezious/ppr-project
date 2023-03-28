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

std::atomic<bool> running(true);

void signal_handler(int sig)
{
  if (sig == 1 || sig == 2 || sig == 3)
  {
    running = false;
  }
}

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

int determineThreadCount(const int processorCount, const int resultMatrixRows)
{
  if (resultMatrixRows <= processorCount)
  {
    return resultMatrixRows;
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

int main(void)
{
  srand((unsigned int)time(NULL));
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = signal_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  int firstMatrixRows = 0;
  int firstMatrixColumns = 0;
  int secondMatrixRows = 0;
  int secondMatrixColumns = 0;

  std::cout << "Enter the number of rows for the first matrix: " << std::endl;
  std::cin >> firstMatrixRows;

  std::cout << "Enter the number of columns for the first matrix:  "
            << std::endl;
  std::cin >> firstMatrixColumns;

  std::cout << "Enter the number of columns for the second matrix: "
            << std::endl;
  std::cin >> secondMatrixColumns;

  secondMatrixRows = firstMatrixColumns;

  int **firstMatrix = getMatrix(firstMatrixRows, firstMatrixColumns);
  int **secondMatrix = getMatrix(secondMatrixRows, secondMatrixColumns);

  std::cout << "Matrix A: \n";
  printMatrix(firstMatrixRows, firstMatrixColumns, firstMatrix);
  std::cout << "Matrix B: \n";
  printMatrix(secondMatrixRows, secondMatrixColumns, secondMatrix);

  int **resultMatrix = getMatrix(firstMatrixRows, secondMatrixColumns, false);

  const auto processor_count = omp_get_num_procs();
  const int thread_count =
      determineThreadCount(processor_count, firstMatrixRows);
  omp_set_num_threads(thread_count + 1); // +1 is necessary to get desired amount

  const std::vector<std::pair<int, int>> rowStartRowEnd =
      determineRowDistribution(thread_count, firstMatrixRows);

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    if (id)
    {
      const int startRow = rowStartRowEnd.at(id - 1).first; // id starts with 1
      const int endRow = rowStartRowEnd.at(id - 1).second;

      for (int currentRow = startRow; currentRow <= endRow; ++currentRow)
      {
        for (int c2 = 0; c2 < secondMatrixColumns; ++c2)
        {
          int result = 0;
          for (int r2 = 0; r2 < secondMatrixRows; ++r2)
          {
            int mult = firstMatrix[currentRow][r2] * secondMatrix[r2][c2];
            result += mult;
          }
          resultMatrix[currentRow][c2] = result;
        }
      }
    }
  }

  std::cout << "Matrix AB: \n";
  printMatrix(firstMatrixRows, secondMatrixColumns, resultMatrix);

  for (int h = 0; h < firstMatrixRows; h++)
  {
    delete[] firstMatrix[h];
  }
  delete[] firstMatrix;
  for (int h = 0; h < secondMatrixRows; h++)
  {
    delete[] secondMatrix[h];
  }
  delete[] secondMatrix;

  for (int h = 0; h < secondMatrixRows; h++)
  {
    delete[] resultMatrix[h];
  }
  delete[] resultMatrix;

  return 0;
}