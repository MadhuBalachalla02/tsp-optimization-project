#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>


using namespace std;

// Define constants and TSP_State structure here (as discussed previously)
// NOTE: Assuming all constants and the TSP_State struct are defined here.
// e.g., const double LOCAL_RUN_TIME_LIMIT = 60.0;
// e.g., struct TSP_State { ... };

struct RunResult {
  std::vector<int> tour;
  double cost;
  long long total_moves;
  int iteration_count;
};

// ------------------------------------

// --- NEW STATE STRUCTURE ---
struct TSP_State {
  vector<vector<double>> D;          // Distance matrix
  int N = 0;                         // Number of nodes
  vector<vector<int>> nn_list;       // Nearest Neighbor list
  long long total_moves_checked = 0; // Move counter

  // FIX: Add the 'mutable' keyword here
  mutable std::mt19937 generator; // Random generator

  const int K_NN_LIST;

  // Constructor to initialize constants and the generator
  TSP_State(int k_nn) : K_NN_LIST(k_nn) {
    // Initialize generator once on creation
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  // A method to reset state for a new run if needed (though not strictly
  // necessary if a new struct instance is created for each run)
  void reset_counters() { total_moves_checked = 0; }
};

// Global Constants can remain, or be moved inside the struct
const double MAX_RUNTIME_SECONDS = 120.0;
const double GLOBAL_TIME_LIMIT_SAFETY = MAX_RUNTIME_SECONDS - 2.0;
const int K_NN_LIST_GLOBAL = 75; // Use this to initialize the struct
const string DEFAULT_SOLUTION_FILENAME = "solution_915552766.txt";
const double LOCAL_RUN_TIME_LIMIT = 60.0; // <-- ADD THIS LINE HERE

void appendTourToFile(const std::vector<int> &tour,
                      const std::string &filename) {
  // Use std::ios::app to append to the file
  std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open/create output file " << filename
              << std::endl;
    return;
  }

  // Output sequence of node indices separated by commas (1-based indices)
  for (size_t i = 0; i < tour.size(); ++i) {
    // Node indices are 0-based, so add 1 for output
    outfile << tour[i] + 1;
    if (i < tour.size() - 1) {
      outfile << ", ";
    }
  }
  // Add the return edge (closing the cycle)
  outfile << ", " << tour.front() + 1;

  // Add a newline to separate the current tour from the next one
  outfile << "\n";

  outfile.close();
}

/**
 * @brief Outputs the final tour to the required text file format.
 * @param final_tour The best tour found by the algorithm (0-based indices).
 * @param filename The path and name of the file to write to.
 */
void outputSolutionToFile(const vector<int> &final_tour,
                          const string &filename = DEFAULT_SOLUTION_FILENAME) {
  std::ofstream outfile(filename); // Use the provided filename
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not create output file " << filename
              << std::endl;
    return;
  }

  // Output sequence of node indices separated by commas (1-based indices)
  for (size_t i = 0; i < final_tour.size(); ++i) {
    // Node indices are 0-based, so add 1 for output
    outfile << final_tour[i] + 1;
    if (i < final_tour.size() - 1) {
      outfile << ", ";
    }
  }
  // Add the return edge (closing the cycle)
  outfile << ", " << final_tour.front() + 1;

  outfile.close();
  cerr << "Solution written to: " << filename << endl;
}

// ----------------------------------------------------
// CORE HELPERS
// ----------------------------------------------------

// Calculates the cost of the full tour (must be defined by the user)
/**
 * @brief Calculates the cost of the full tour using the state's distance
 * matrix.
 * @param tour The sequence of node indices (0-based).
 * @param state A constant reference to the current TSP_State object.
 * @return The total cost of the tour.
 */
double calculateTourCost(const vector<int> &tour, const TSP_State &state) {
  double cost = 0.0;

  // Sum the cost of all internal edges
  for (size_t i = 0; i < tour.size() - 1; ++i) {
    // Access the distance matrix via the state object
    cost += state.D[tour[i]][tour[i + 1]];
  }

  // Add the return edge to close the cycle (last node to first node)
  // Access the distance matrix via the state object
  cost += state.D[tour.back()][tour.front()];

  return cost;
}

//***
//* @brief Constructs an initial tour using the Nearest Neighbor heuristic.
// @return vector<int> The initial tour (sequence of node indices).

/**
 * @brief Constructs an initial tour using the Nearest Neighbor heuristic.
 * @param state A constant reference to the current TSP_State object.
 * @return vector<int> The initial tour (sequence of node indices).
 */
vector<int> nearestNeighborTour(const TSP_State &state) {
  // Use state.N for size initialization
  vector<int> tour;
  vector<bool> visited(state.N, false);

  // Start at Node 0 (arbitrary starting point)
  int current_node = 0;
  visited[current_node] = true;
  tour.push_back(current_node);

  // Loop N-1 times to visit the remaining nodes
  for (int i = 0; i < state.N - 1; ++i) { // Use state.N
    double min_dist = numeric_limits<double>::max();
    int next_node = -1;

    // Find the nearest unvisited neighbor
    for (int neighbor = 0; neighbor < state.N; ++neighbor) { // Use state.N
      // Check if unvisited AND closer than the current minimum
      // Access distance matrix via state.D
      if (!visited[neighbor] && state.D[current_node][neighbor] < min_dist) {
        min_dist = state.D[current_node][neighbor];
        next_node = neighbor;
      }
    }

    // If a valid next node was found
    if (next_node != -1) {
      current_node = next_node;
      visited[current_node] = true;
      tour.push_back(current_node);
    }
  }
  return tour;
}

//**
//* @brief Generates a new random starting tour by shuffling the node indices.
//* @param state A constant reference to the current TSP_State object.
//* @return vector<int> The randomly generated tour.

vector<int> generateRandomTour(const TSP_State &state) {
  // 1. Use state.N to size the tour vector
  vector<int> tour(state.N);

  // std::iota fills the vector with 0, 1, 2, ..., N-1
  std::iota(tour.begin(), tour.end(), 0);

  // 2. Shuffle the tour randomly using the generator from the state object
  // Note: std::shuffle takes the generator by reference, allowing it to modify
  // the generator's internal state.
  std::shuffle(tour.begin(), tour.end(), state.generator);

  return tour;
}

// ----------------------------------------------------
// NN PRUNING LOGIC
// ----------------------------------------------------

/**
 * @brief Precomputes the K_NN_LIST closest neighbors for every node and stores
 * it in the state.
 * @param state A reference to the current TSP_State object (modified by this
 * function).
 */
void precomputeNNList(TSP_State &state) {
  // 1. Assign the list directly to the state member.
  // Use state.N and state.K_NN_LIST
  state.nn_list.assign(state.N, vector<int>(state.K_NN_LIST));

  for (int i = 0; i < state.N; ++i) { // Use state.N
    // Create a list of all nodes and their distance to node i
    vector<pair<double, int>> distances;
    for (int j = 0; j < state.N; ++j) { // Use state.N
      if (i != j) {
        // Access distance matrix via state.D
        distances.push_back({state.D[i][j], j});
      }
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end());

    // Store the top K neighbors
    int limit =
        std::min((int)distances.size(), state.K_NN_LIST); // Use state.K_NN_LIST
    for (int k = 0; k < limit; ++k) {
      // Store results in state.nn_list
      state.nn_list[i][k] = distances[k].second;
    }
  }
  // Update cerr output
  cerr << "Precomputed " << state.K_NN_LIST << "-NN list for all " << state.N
       << " nodes." << endl;
}

// ----------------------------------------------------
// 1. O(N^2) UNCONSTRAINED 2-OPT SWAP
// ----------------------------------------------------

/**
 * @brief Performs a First-Improvement UNCONSTRAINED 2-Opt local search
 * (O(N^2)). The search is terminated early if the global time limit is reached.
 * @param tour The initial tour to improve.
 * @param start_time The global starting time point for time limit enforcement.
 * @return The locally optimized tour.
 */
/**
 * @brief Performs the 2-Opt local search until a local minimum is found or time
 * limits are exceeded.
 * @param tour The input tour to optimize.
 * @param global_start_time The start time of the entire program (for the
 * GLOBAL_TIME_LIMIT_SAFETY check).
 * @param local_start_time The start time of the current multi-start run (for
 * the LOCAL_RUN_TIME_LIMIT check).
 * @param state A reference to the current TSP_State object (modified to update
 * total_moves_checked).
 * @return vector<int> The locally optimized tour.
 */
vector<int>
twoOpt(vector<int> tour,
       const std::chrono::steady_clock::time_point &global_start_time,
       const std::chrono::steady_clock::time_point &local_start_time,
       TSP_State &state) { // Pass by non-const reference to update counters

  bool improved = true;
  int N_nodes = tour.size();

  while (improved) {
    improved = false;
    bool move_was_made = false;

    // --- GLOBAL SAFETY CHECK (Program Timer) ---
    std::chrono::duration<double> total_duration =
        std::chrono::steady_clock::now() - global_start_time;
    // Use state.GLOBAL_TIME_LIMIT_SAFETY or the global constant if you kept it
    // outside the struct
    if (total_duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
      return tour;
    }

    // --- LOCAL RUN CHECK (1-Minute Timer) ---
    // Use the global constant LOCAL_RUN_TIME_LIMIT defined outside the function
    std::chrono::duration<double> run_duration =
        std::chrono::steady_clock::now() - local_start_time;
    if (run_duration.count() >= LOCAL_RUN_TIME_LIMIT) {
      return tour;
    }

    // Iterate over the first cut point 'i'
    for (int i = 0; i < N_nodes; ++i) {
      int i1 = (i + 1) % N_nodes;

      // Iterate over the second cut point 'k'
      for (int k = 0; k < N_nodes; ++k) {

        int k1 = (k + 1) % N_nodes;

        if (k == i || k == i1 || k1 == i || k1 == i1)
          continue;

        // Update the counter inside the state structure
        state.total_moves_checked++;

        // Edge removal: (u, v) and (x, y)
        int u = tour[i];
        int v = tour[i1];
        int x = tour[k];
        int y = tour[k1];

        // Access distance matrix via state.D
        double removed_cost = state.D[u][v] + state.D[x][y];
        double added_cost = state.D[u][x] + state.D[v][y];

        double gain = removed_cost - added_cost;

        if (gain > 1e-9) {
          // Found a profitable 2-Opt move. Execute the segment reversal.
          // ... (Segment reversal logic remains the same) ...

          if (i1 < k) {
            std::reverse(tour.begin() + i1, tour.begin() + k + 1);
          } else {
            // Wrapping segment reversal: (i1 -> ... -> N-1 -> 0 -> ... -> k)
            vector<int> temp_segment;
            temp_segment.insert(temp_segment.end(), tour.begin() + i1,
                                tour.end());
            temp_segment.insert(temp_segment.end(), tour.begin(),
                                tour.begin() + k + 1);

            std::reverse(temp_segment.begin(), temp_segment.end());

            int segment_size_part1 = N_nodes - i1;
            std::copy(temp_segment.begin(),
                      temp_segment.begin() + segment_size_part1,
                      tour.begin() + i1);
            std::copy(temp_segment.begin() + segment_size_part1,
                      temp_segment.end(), tour.begin());
          }

          improved = true;
          move_was_made = true;
          // Break out of inner loops and restart the while loop
          // (First-Improvement)
          goto found_improvement;
        }
      }
    }

  found_improvement:;
    if (move_was_made)
      continue;

    improved = false;
  }
  return tour;
}

// ----------------------------------------------------
// 2. O(N*K) CONSTRAINED RELOCATE-OPT (L=1 Or-Opt)
// ----------------------------------------------------
/**
 * @brief Performs a First-Improvement Relocate-Opt (1-Node Or-Opt) local
 * search. The reinsertion point is constrained by the NN list of the preceding
 * node (O(N*K)).
 * @param tour The initial tour to improve.
 * @param start_time The global starting time point for time limit enforcement.
 * @return The locally optimized tour.
 */
/**
 * @brief Performs the Relocate (1-Opt) local search, optimized using the
 * Nearest Neighbor list.
 * @param tour The input tour to optimize.
 * @param global_start_time The start time of the entire program (for the
 * GLOBAL_TIME_LIMIT_SAFETY check).
 * @param local_start_time The start time of the current multi-start run (for
 * the LOCAL_RUN_TIME_LIMIT check).
 * @param state A reference to the current TSP_State object (modified to update
 * total_moves_checked).
 * @return vector<int> The locally optimized tour.
 */
vector<int>
relocateOpt(vector<int> tour,
            const std::chrono::steady_clock::time_point &global_start_time,
            const std::chrono::steady_clock::time_point &local_start_time,
            TSP_State &state) { // Added TSP_State& state

  bool improved = true;
  int N_nodes = tour.size();

  // Map to quickly find the index of a node (ID) in the tour (index)
  vector<int> node_to_index(N_nodes);
  auto update_node_to_index = [&]() {
    for (int idx = 0; idx < N_nodes; ++idx) {
      node_to_index[tour[idx]] = idx;
    }
  };
  update_node_to_index(); // Initialize map

  while (improved) {
    improved = false;
    bool move_was_made = false;

    // --- GLOBAL SAFETY CHECK (Program Timer) ---
    std::chrono::duration<double> total_duration =
        std::chrono::steady_clock::now() - global_start_time;
    if (total_duration.count() >=
        GLOBAL_TIME_LIMIT_SAFETY) { // Uses global constant
      return tour;
    }

    // --- NEW LOCAL RUN CHECK (1-Minute Timer) ---
    std::chrono::duration<double> run_duration =
        std::chrono::steady_clock::now() - local_start_time;
    if (run_duration.count() >= LOCAL_RUN_TIME_LIMIT) { // Uses global constant
      return tour;
    }

    // Iterate over every node i (the node to be relocated).
    for (int i = 0; i < N_nodes; ++i) {

      // Nodes involved in the segment removal:
      int u = tour[(i - 1 + N_nodes) % N_nodes]; // Node before i (u)
      int v = tour[i];                           // Node i (v)
      int w = tour[(i + 1) % N_nodes];           // Node after i (w)

      // Removed edges: (u, v) and (v, w)
      // Access distance matrix via state.D
      double removed_cost = state.D[u][v] + state.D[v][w];

      // The segment is removed and replaced by new edge (u, w).
      // double cost_after_remove = state.D[u][w]; // This is redundant, but
      // leaving the calculation structure below

      // Iterate over the K Nearest Neighbors of node 'u'.
      // FIX: Use state.nn_list instead of nn_list_e
      for (int p_node : state.nn_list[u]) {

        int p_idx = node_to_index[p_node];

        // Nodes involved in the reinsertion:
        int p = tour[p_idx];                 // Insertion predecessor (p)
        int q = tour[(p_idx + 1) % N_nodes]; // Insertion successor (q)

        // Skip adjacent or identity moves
        if (p == v || q == v || p == u || q == u)
          continue;

        // Update the counter inside the state structure
        state.total_moves_checked++;

        // New edges: (p, v) and (v, q)
        // Access distance matrix via state.D
        double old_edges_insert = state.D[p][q]; // Edge broken by insertion
        double new_edges_insert =
            state.D[p][v] + state.D[v][q]; // Edges created by insertion

        // Total gain calculation:
        // Gain = (Removed Edges) - (New Edges)
        double gain = (removed_cost + old_edges_insert) -
                      (state.D[u][w] + new_edges_insert);

        if (gain > 1e-9) {
          // *** FIRST-IMPROVEMENT: Execute the move immediately ***

          // 1. Remove node v from its current position
          int current_i = node_to_index[v];
          tour.erase(tour.begin() + current_i);

          // 2. Calculate the new insertion index.
          int new_p_idx = (current_i < p_idx) ? p_idx - 1 : p_idx;

          // 3. Insert node v at the new position (after p_node)
          tour.insert(tour.begin() + new_p_idx + 1, v);

          // Update index map and restart the loop
          update_node_to_index();
          improved = true;
          move_was_made = true;
          goto found_improvement;
        }
      }
    }

  found_improvement:;
    if (move_was_made)
      continue;

    improved = false;
  }
  return tour;
}

// --- Modified readGraph function to include a debug counter ---

/**
 * @brief Reads graph data (N and distance matrix D) from a file and stores it
 * in the state.
 * @param filename The path and name of the input file.
 * @param state A reference to the current TSP_State object (modified by this
 * function).
 */
void readGraph(const std::string &filename, TSP_State &state) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    exit(1);
  }

  // Read N and assign it to state.N
  if (!(file >> state.N)) {
    std::cerr << "Error reading number of nodes (N)." << std::endl;
    exit(1);
  }

  // --- CORRECT HEADER SKIP ---
  // 1. Read and discard the rest of the line containing 'N'.
  std::string header_line;
  std::getline(file, header_line);

  // 2. Read and discard the header line ("Node1 Node2 Distance").
  std::getline(file, header_line);
  // ---------------------------

  // Assign the distance matrix D to state.D, resizing based on state.N
  state.D.assign(state.N, std::vector<double>(state.N, 0.0));

  int u, v;
  double dist;
  int line_count = 0;

  while (file >> u >> v >> dist) {
    // Adjust to 0-based indexing
    u--;
    v--;

    // Only process if indices are valid and it's not a self-loop
    if (u >= 0 && u < state.N && v >= 0 && v < state.N &&
        u != v) { // Use state.N
      // Assign distances to state.D
      state.D[u][v] = dist;
      state.D[v][u] = dist;
      line_count++;
    }
  }

  file.close();
  std::cerr << "DEBUG: Total edges read and processed: " << line_count
            << std::endl;
}
/**
 * @brief Constructs an initial tour using the Nearest Neighbor heuristic.
 * @return vector<int> The initial tour (sequence of node indices).
 */

// NOTE: Assuming TSP_State and RunResult structs, and all helper functions
// (readGraph, precomputeNNList, twoOpt, relocateOpt, etc.) are defined with the
// refactored signatures.

// Global constant definition (needed for the local time limit check)

/**
 * @brief Executes the Iterated Local Search (ILS) optimization for one graph
 * file while respecting the 1-minute local time limit.
 * @param filename The name of the input graph file.
 * @param global_start_time The start time of the entire program (for global
 * safety checks).
 * @return RunResult Struct containing the best tour, cost, total moves, and
 * iterations.
 */
RunResult run_optimization_scenario(
    const std::string &filename,
    const std::chrono::steady_clock::time_point &global_start_time) {

  // --- 1. INITIALIZE CLEAN STATE FOR THIS RUN ---
  // K_NN_LIST_GLOBAL should be defined outside (e.g., const int
  // K_NN_LIST_GLOBAL = 75;)
  TSP_State state(K_NN_LIST_GLOBAL);

  // 2. Read Graph Data (modifies state.N and state.D)
  readGraph(filename, state);
  std::cerr << "Nodes (N): " << state.N << std::endl;

  // 3. Precompute Nearest Neighbor List (modifies state.nn_list)
  precomputeNNList(state);

  // Initialize best tour (using refactored nearestNeighborTour)
  std::vector<int> best_tour = nearestNeighborTour(state);
  double best_cost =
      calculateTourCost(best_tour, state); // Use refactored cost function
  std::cerr << "Initial Cost (NN Start): " << std::fixed << best_cost
            << std::endl;

  int iteration_count = 1;

  // -------------------------------------------------
  // ILS MAIN LOOP (Restructured from your old main)
  // -------------------------------------------------

  // --- 4. LOCAL TIME CHECK ---
  // Capture the start time for THIS specific run/iteration
  auto local_start_time = std::chrono::steady_clock::now();

  // Loop until the global safety limit OR the 1-minute local limit is hit.
  while (true) {

    // Capture current time
    auto current_time = std::chrono::steady_clock::now();

    // Global safety check (program must stop if this limit is hit)
    std::chrono::duration<double> total_duration =
        current_time - global_start_time;
    if (total_duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
      std::cerr << "INFO: Global program time limit reached." << std::endl;
      break;
    }

    // Local run check (This run must stop if 60 seconds are up)
    if (std::chrono::duration<double>(current_time - local_start_time)
            .count() >= LOCAL_RUN_TIME_LIMIT) {
      // Break if we exceeded the time limit for this graph
      break;
    }

    iteration_count++;
    std::vector<int> current_tour = best_tour;

    // --- 5. PERTURBATION STRATEGY (Your original logic) ---
    if (iteration_count > 1) {
      std::uniform_int_distribution<> index_dist(0, state.N - 1);
      int start_idx = index_dist(state.generator); // Use state.generator

      int min_len = std::min(state.N / 20, 10);
      int max_len = std::min(state.N / 5, 100);
      if (max_len < min_len)
        max_len = min_len + 1;

      // Use state.generator
      std::uniform_int_distribution<> len_dist(min_len, max_len);
      int segment_length = len_dist(state.generator);
      int end_idx = (start_idx + segment_length) % state.N;

      if (start_idx > end_idx)
        std::swap(start_idx, end_idx);

      std::reverse(current_tour.begin() + start_idx,
                   current_tour.begin() + end_idx);
    }

    // --- 6. CHAINED LOCAL SEARCH (Requires passing state and both timers) ---

    // Pass global_start_time, local_start_time, AND the state reference
    current_tour =
        twoOpt(current_tour, global_start_time, local_start_time, state);

    // Pass global_start_time, local_start_time, AND the state reference
    std::vector<int> local_optimized_tour =
        relocateOpt(current_tour, global_start_time, local_start_time, state);

    // -----------------------------

    // 7. ACCEPTANCE CRITERION
    double local_cost = calculateTourCost(
        local_optimized_tour, state); // Use refactored cost function

    // Check for global improvement
    if (local_cost < best_cost) {
      best_cost = local_cost;
      best_tour = local_optimized_tour;
      std::cerr << "--- New Best Cost (" << iteration_count
                << "): " << std::fixed << best_cost << std::endl;
    }

    // Explicitly check the local time limit here before continuing to the next
    // iteration
    std::chrono::duration<double> current_run_time =
        std::chrono::steady_clock::now() - local_start_time;
    if (current_run_time.count() >= LOCAL_RUN_TIME_LIMIT) {
      std::cerr << "INFO: Local 1-minute time limit reached for this graph."
                << std::endl;
      break;
    }
  }

  // --- 8. PREPARE AND RETURN RESULTS ---
  RunResult final_result;
  final_result.tour = best_tour;
  final_result.cost = best_cost;
  final_result.total_moves =
      state.total_moves_checked; // Retrieve final counter value
  final_result.iteration_count = iteration_count;

  std::cerr << "Final Cost: " << std::fixed << std::setprecision(2) << best_cost
            << std::endl;
  std::cerr << "Total Cycles Checked: " << std::scientific
            << std::setprecision(1) << (double)final_result.total_moves
            << std::endl;
  std::cerr << "Total Iterations: " << final_result.iteration_count
            << std::endl;

  // When this function returns, the local 'state' object is destroyed, clearing
  // D, nn_list, etc.
  return final_result;
}

// ----------------------------------------------------
// ----------------------------------------------------

int main(int argc, char *argv[]) {
  // Argument check (allowing hardcoded files if no arguments are given)
  if (argc != 1 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " (or <input_file_1> <input_file_2>)"
              << std::endl;
  }

  // Define the file names
  // If running from command line: use arguments. Otherwise, use hardcoded
  // names.
  std::string filename_E =
      (argc == 3) ? argv[1] : "TSP_1000_euclidianDistance.txt";
  std::string filename_R =
      (argc == 3) ? argv[2] : "TSP_1000_randomDistance.txt";
  const std::string OUTPUT_FILENAME = "solutions_915552766.txt";

  // IMPORTANT: Clear the output file before starting the first run.
  std::ofstream(OUTPUT_FILENAME, std::ios::trunc).close();

  // This time point tracks the absolute start of the program (for the 60s
  // safety check).
  auto global_start_time = std::chrono::steady_clock::now();

  RunResult result1;
  RunResult result2;

  // --- SCENARIO 1: EUCLIDIAN GRAPH (1-Minute Limit) ---
  std::cerr << "\n=================================================="
            << std::endl;
  std::cerr << "Starting Optimization for Graph 1: " << filename_E
            << " (1-Minute Limit)" << std::endl;
  std::cerr << "=================================================="
            << std::endl;

  // Execute the optimization scenario and store results
  // (Assuming run_optimization_scenario returns a RunResult struct)
  result1 = run_optimization_scenario(filename_E, global_start_time);

  // Append the first tour to the combined output file
  appendTourToFile(result1.tour, OUTPUT_FILENAME);

  // --- SCENARIO 2: RANDOM GRAPH (1-Minute Limit) ---
  std::cerr << "\n=================================================="
            << std::endl;
  std::cerr << "Starting Optimization for Graph 2: " << filename_R
            << " (1-Minute Limit)" << std::endl;
  std::cerr << "=================================================="
            << std::endl;

  // Execute the second, independent optimization scenario
  result2 = run_optimization_scenario(filename_R, global_start_time);

  // Append the second tour to the combined output file (on the next line)
  appendTourToFile(result2.tour, OUTPUT_FILENAME);

  // -------------------------------------------------
  // FINAL CONSOLE OUTPUT (Scientific Format)
  // -------------------------------------------------
  std::cout << "\n--- RESULTS SUMMARY ---" << std::endl;
  std::cout << std::scientific << std::setprecision(1);

  // Graph 1 Output
  std::cout << "Graph 1 Cost: " << result1.cost << std::endl;
  std::cout << "Graph 1 Cycles: " << (double)result1.total_moves << std::endl;

  // Graph 2 Output
  std::cout << "Graph 2 Cost: " << result2.cost << std::endl;
  std::cout << "Graph 2 Cycles: " << (double)result2.total_moves << std::endl;

  std::cerr << "\n--- Global Program Summary ---" << std::endl;
  std::cerr << "Combined Solutions written to: " << OUTPUT_FILENAME
            << std::endl;

  return 0;
}