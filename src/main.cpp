#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <signal.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

std::atomic<bool> running(true);

void signal_handler(int sig) {
  if (sig == 1 || sig == 2 || sig == 3) {
    running = false;
  }
}

int **getMatrix(int rows, int columns) {
  srand(time(0));
  int **array2D = 0;
  array2D = new int *[rows];

  for (int r = 0; r < rows; r++) {
    array2D[r] = new int[columns];

    for (int c = 0; c < columns; c++) {
      array2D[r][c] = rand() % 100000;
    }
  }
  return array2D;
}

int determineThreadCount(const int processorCount, const int resultMatrixRows) {
  if (resultMatrixRows <= processorCount) {
    return resultMatrixRows;
  }
  return processorCount;
}

std::vector<std::pair<int, int>> determineRowDistribution(const int threads,
                                                          const int tasks) {
  std::vector<std::pair<int, int>> rowStartRowEnd;
  int index = 0;
  for (int i = 0; i < threads; ++i) {
    int tasks_for_this_thread = tasks / threads + (i < tasks % threads);
    rowStartRowEnd.push_back(std::make_pair(index, index + tasks_for_this_thread - 1));
    index += tasks_for_this_thread;
  }
  return rowStartRowEnd;
}

int main(void) {
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

  const int **firstMatrix = getMatrix(firstMatrixRows, firstMatrixColumns);
  const int **secondMatrix = getMatrix(secondMatrixRows, secondMatrixColumns);
  int resultMatrix[firstMatrixRows][secondMatrixColumns];

  const auto processor_count = omp_get_num_procs();
  const int thread_count =
      determineThreadCount(processor_count, firstMatrixRows);
    omp_set_num_threads(thread_count);
  const std::vector<std::pair<int, int>> rowStartRowEnd =
      determineRowDistribution(thread_count, firstMatrixRows);

  for (const std::pair<int, int> &p : rowStartRowEnd) {
    std::cout << p.first << " " << p.second << std::endl;
  }

  #pragma omp parallel
    {
        int id = omp_get_thread_num();
        std::cout << id;
    }

  for (int h = 0; h < firstMatrixRows; h++) {
    delete[] firstMatrix[h];
  }
    delete[] firstMatrix;
  for (int h = 0; h < secondMatrixRows; h++) {
    delete[] secondMatrix[h];
  }
  delete[] secondMatrix;

  return 0;
}