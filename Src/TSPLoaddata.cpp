#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <tuple> 

using namespace std;

// --- GLOBAL CONSTANTS (TUNED FOR 60 SECONDS) ---
const double MAX_RUNTIME_SECONDS = 60.0;
const double GLOBAL_TIME_LIMIT_SAFETY = MAX_RUNTIME_SECONDS - 0.5; // Stop at 58.0 seconds
// The number of nearest neighbors to consider for any move (LKH standard is 15-20)
const int K_NN_LIST = 75; 
// --- Global Data Structures

vector<vector<double>> D; // Distance matrix
int N; // Number of nodes
// Stores the K_NN_LIST closest neighbor indices for each node.
vector<vector<int>> nn_list; 


// ----------------------------------------------------
// GLOBAL COUNTER AND RANDOM GENERATOR
// ----------------------------------------------------
static long long total_moves_checked = 0; 
static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count()); 
//static std::mt19937 generator(42); 
// Define a default filename, which can be overridden by the function argument.
const string DEFAULT_SOLUTION_FILENAME = "solution_915552766.txt"; 

/**
 * @brief Outputs the final tour to the required text file format.
 * @param final_tour The best tour found by the algorithm (0-based indices).
 * @param filename The path and name of the file to write to.
 */
void outputSolutionToFile(const vector<int>& final_tour, const string& filename = DEFAULT_SOLUTION_FILENAME) {
    std::ofstream outfile(filename); // Use the provided filename
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file " << filename << std::endl;
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
double calculateTourCost(const vector<int>& tour) {
    double cost = 0.0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        cost += D[tour[i]][tour[i+1]];
    }
    // Add return edge to close the cycle
    cost += D[tour.back()][tour.front()]; 
    return cost;
}

//*** 
 //* @brief Constructs an initial tour using the Nearest Neighbor heuristic.
 // @return vector<int> The initial tour (sequence of node indices).

vector<int> nearestNeighborTour() {
    vector<int> tour;
    vector<bool> visited(N, false);
    
    // Start at Node 0 (Node 1 in the file format—an arbitrary starting point)
    int current_node = 0;
    visited[current_node] = true;
    tour.push_back(current_node);

    // Loop N-1 times to visit the remaining 999 nodes
    for (int i = 0; i < N - 1; ++i) {
        double min_dist = numeric_limits<double>::max();
        int next_node = -1;

        // Find the nearest unvisited neighbor
        for (int neighbor = 0; neighbor < N; ++neighbor) {
            // Check if the neighbor is unvisited AND closer than the current minimum
            if (!visited[neighbor] && D[current_node][neighbor] < min_dist) {
                min_dist = D[current_node][neighbor];
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


/**
 * @brief Generates a new random starting tour by shuffling the node indices.
 */
vector<int> generateRandomTour() {
    vector<int> tour(N);
    std::iota(tour.begin(), tour.end(), 0);
    // Shuffle the tour randomly
    std::shuffle(tour.begin(), tour.end(), generator); 
    return tour;
}

// ----------------------------------------------------
// NN PRUNING LOGIC
// ----------------------------------------------------

/**
 * @brief Precomputes the K_NN_LIST closest neighbors for every node.
 */
void precomputeNNList() {
    nn_list.assign(N, vector<int>(K_NN_LIST));
    for (int i = 0; i < N; ++i) {
        // Create a list of all nodes and their distance to node i
        vector<pair<double, int>> distances;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                distances.push_back({D[i][j], j});
            }
        }
        
        // Sort by distance
        std::sort(distances.begin(), distances.end());
        
        // Store the top K neighbors
        int limit = std::min((int)distances.size(), K_NN_LIST);
        for (int k = 0; k < limit; ++k) {
            nn_list[i][k] = distances[k].second;
        }
    }
    cerr << "Precomputed " << K_NN_LIST << "-NN list for all " << N << " nodes." << endl;
}

// ----------------------------------------------------
// 1. O(N^2) UNCONSTRAINED 2-OPT SWAP
// ----------------------------------------------------

/**
 * @brief Performs a First-Improvement UNCONSTRAINED 2-Opt local search (O(N^2)).
 * The search is terminated early if the global time limit is reached.
 * @param tour The initial tour to improve.
 * @param start_time The global starting time point for time limit enforcement.
 * @return The locally optimized tour.
 */
vector<int> twoOpt(vector<int> tour, const std::chrono::steady_clock::time_point& start_time) {
    bool improved = true;
    int N_nodes = tour.size();
    
    while (improved) {
        improved = false;
        bool move_was_made = false;

        // Safety Time check (Global 59.0s limit)
        std::chrono::duration<double> total_duration = std::chrono::steady_clock::now() - start_time;
        if (total_duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
            return tour;
        }
        
        // Iterate over the first cut point 'i' (N iterations)
        for (int i = 0; i < N_nodes; ++i) {
            int i1 = (i + 1) % N_nodes; // Index after i
            
            // Iterate over the second cut point 'k' (N iterations)
            for (int k = 0; k < N_nodes; ++k) {
                
                int k1 = (k + 1) % N_nodes; // Index after k
                
                // Ensure the two edges (i, i1) and (k, k1) are not adjacent or identical.
                // i != k, i != k1, i1 != k, i1 != k1.
                if (k == i || k == i1 || k1 == i || k1 == i1) continue; 
                
                total_moves_checked++; 

                // Edge removal: (u, v) and (x, y)
                int u = tour[i];    // Node i
                int v = tour[i1];   // Node i+1
                int x = tour[k];    // Node k
                int y = tour[k1];   // Node k+1

                double removed_cost = D[u][v] + D[x][y];
                // Edges added: (u, x) and (v, y)
                double added_cost = D[u][x] + D[v][y];

                double gain = removed_cost - added_cost;

                if (gain > 1e-9) { 
                    // Found a profitable 2-Opt move. Execute the segment reversal.

                    // Check for normal (non-wrapping) segment: [i1, k]
                    if (i1 < k) {
                        std::reverse(tour.begin() + i1, tour.begin() + k + 1);
                    } else {
                        // Wrapping segment reversal: Reversing the path from i1 -> ... -> N-1 -> 0 -> ... -> k
                        vector<int> temp_segment;
                        // Segment 1: [i+1, N-1]
                        temp_segment.insert(temp_segment.end(), tour.begin() + i1, tour.end());
                        // Segment 2: [0, k]
                        temp_segment.insert(temp_segment.end(), tour.begin(), tour.begin() + k + 1);

                        std::reverse(temp_segment.begin(), temp_segment.end());

                        // Reassemble the tour
                        // The segment size is (N - i1) + (k + 1)
                        int segment_size_part1 = N_nodes - i1;
                        std::copy(temp_segment.begin(), temp_segment.begin() + segment_size_part1, tour.begin() + i1);
                        std::copy(temp_segment.begin() + segment_size_part1, temp_segment.end(), tour.begin());
                    }
                    
                    improved = true; 
                    move_was_made = true;
                    // Break out of inner loops and restart the while loop (First-Improvement)
                    goto found_improvement;
                }
            }
        }
        
        found_improvement:; 
        if (move_was_made) continue; 
        
        improved = false;
    }
    return tour;
}


// ----------------------------------------------------
// 2. O(N*K) CONSTRAINED RELOCATE-OPT (L=1 Or-Opt)
// ----------------------------------------------------
/**
 * @brief Performs a First-Improvement Relocate-Opt (1-Node Or-Opt) local search. 
 * The reinsertion point is constrained by the NN list of the preceding node (O(N*K)).
 * @param tour The initial tour to improve.
 * @param start_time The global starting time point for time limit enforcement.
 * @return The locally optimized tour.
 */
vector<int> relocateOpt(vector<int> tour, const std::chrono::steady_clock::time_point& start_time) {
    bool improved = true;
    int N_nodes = tour.size();
    
    // Map to quickly find the index of a node (ID) in the tour (index)
    vector<int> node_to_index(N_nodes);
    auto update_node_to_index = [&]() {
        for(int idx = 0; idx < N_nodes; ++idx) {
            node_to_index[tour[idx]] = idx;
        }
    };
    update_node_to_index(); // Initialize map

    while (improved) {
        improved = false; 
        bool move_was_made = false;

        // Time check (Global 59.0s limit)
        auto time_check = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = time_check - start_time;
        if (duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
            return tour;
        }
        
        // Iterate over every node i (the node to be relocated). (N iterations)
        for (int i = 0; i < N_nodes; ++i) {
            
            // Nodes involved in the segment removal:
            int u = tour[(i - 1 + N_nodes) % N_nodes]; // Node before i (u)
            int v = tour[i];                          // Node i (v)
            int w = tour[(i + 1) % N_nodes];          // Node after i (w)
            
            // Removed edges: (u, v) and (v, w)
            double removed_cost = D[u][v] + D[v][w];
            
            // The segment is removed and replaced by new edge (u, w).
            double cost_after_remove = D[u][w];
            
            // Iterate over the K Nearest Neighbors of node 'u' (the node that now connects to w).
            // The NN node 'p_node' will be the node *before* the insertion point. (K iterations)
            for (int p_node : nn_list[u]) { 
                
                int p_idx = node_to_index[p_node]; // Index of p_node (insertion predecessor)
                
                // Nodes involved in the reinsertion:
                int p = tour[p_idx]; // Insertion predecessor (p)
                int q = tour[(p_idx + 1) % N_nodes]; // Insertion successor (q)
                
                // Skip if insertion is adjacent to removal, which leads to identity moves or complex wraps
                if (p == v || q == v || p == u || q == u) continue;
                
                total_moves_checked++; 
                
                // New edges: (p, v) and (v, q)
                double old_edges_insert = D[p][q]; // Edge broken by insertion
                double new_edges_insert = D[p][v] + D[v][q]; // Edges created by insertion
                
                // Total gain calculation:
                // Gain = (Removed Edges) - (New Edges)
                double gain = (removed_cost + old_edges_insert) - (D[u][w] + new_edges_insert);

                if (gain > 1e-9) { 
                    // *** FIRST-IMPROVEMENT: Execute the move immediately ***
                    
                    // 1. Remove node v from its current position
                    // Must calculate the current index again as tour might have been changed in the loop
                    int current_i = node_to_index[v]; 
                    tour.erase(tour.begin() + current_i);
                    
                    // 2. Calculate the new insertion index.
                    // If current_i < p_idx, erasing 'i' shifts all indices from current_i+1 to p_idx down by 1.
                    // If current_i > p_idx, erasing 'i' does not affect indices before p_idx.
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
        if (move_was_made) continue; 
        
        improved = false;
    }
    return tour;
}
// --- Modified readGraph function to include a debug counter ---

void readGraph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    if (!(file >> N)) {
        std::cerr << "Error reading number of nodes (N)." << std::endl;
        exit(1);
    }
    
    // --- CORRECT HEADER SKIP ---
    // 1. Read and discard the rest of the line containing 'N' (the newline character).
    std::string header_line;
    std::getline(file, header_line); 
    
    // 2. Read and discard the header line ("Node1 Node2 Distance").
    std::getline(file, header_line); 
    // ---------------------------

    D.assign(N, std::vector<double>(N, 0.0)); 

    int u, v;
    double dist;
    int line_count = 0;

    // --- CLEAN LOOP: Rely on stream for format, as confirmed correct ---
    while (file >> u >> v >> dist) {
        // Adjust to 0-based indexing
        u--; 
        v--;
        
        // Only process if indices are valid and it's not a self-loop (u != v)
        if (u >= 0 && u < N && v >= 0 && v < N && u != v) {
            D[u][v] = dist;
            D[v][u] = dist; 
            line_count++;
        }
    }

    file.close();
    std::cerr << "DEBUG: Total edges read and processed: " << line_count << std::endl;
}

// --- End of readGraph function ---

/**
 * @brief Constructs an initial tour using the Nearest Neighbor heuristic.
 * @return vector<int> The initial tour (sequence of node indices).
 */



// ----------------------------------------------------
// ----------------------------------------------------


int main(int argc, char* argv[]) {
    // ... (argument check and readGraph call) ...
    auto start_time = std::chrono::steady_clock::now(); // This is the START of the entire program (for the final 60s check)

    
    //read file 
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    

    // 1. Read Graph Data
    readGraph(argv[1]);
    cerr << "--- CHECKPOINT 1: Graph Read Complete (" << N << " nodes) ---" << endl; 



    // 2. DEBUG: Print key distances from Node 1 (index 0)
    // We expect these to match the file sample if loading worked.
    if (N > 5) {
        int node_1_index = 0; // Node 1
        int node_2_index = 1; // Node 2
        int node_3_index = 12; // Node 3
        int node_5_index = 44; // Node 5

        cerr << "--- DISTANCE VERIFICATION ---" << endl;
        // Expected: 26.98
        cerr << "D[Node 1][Node 2]: " << fixed << D[node_1_index][node_2_index] << endl;
        // Expected: 53.74
        cerr << "D[Node 1][Node 3]: " << fixed << D[node_1_index][node_3_index] << endl;
        // Expected: 29.43
        cerr << "D[Node 1][Node 5]: " << fixed << D[node_1_index][node_5_index] << endl;
        cerr << "-----------------------------" << endl;
    } else {
        cerr << "DEBUG: Graph too small to run detailed verification check." << endl;
    }
    
     // 0. Precompute Nearest Neighbor List (CRITICAL STEP FOR SPEED)
    precomputeNNList();

    // Initialize best tour with a random tour if not provided by nearestNeighbor()
    vector<int> best_tour= nearestNeighborTour();

    double best_cost = calculateTourCost(best_tour);
    cerr << "--- Initial Cost(NN Start): " << fixed << best_cost << endl;

    int iteration_count = 0;

     // -------------------------------------------------
    // MULTI-START CHAINED LOCAL SEARCH MAIN LOOP 
    // -------------------------------------------------
    while (std::chrono::steady_clock::now() - start_time < std::chrono::duration<double>(GLOBAL_TIME_LIMIT_SAFETY)) {
        
        iteration_count++;
        
        vector<int> current_tour = best_tour; 
        
        // --- PERTURBATION STRATEGY (Random 2-Opt) ---
        if (iteration_count > 1) {
            // Aggressive Perturbation: Reverse a segment of 5% to 20% of the tour length.
            std::uniform_int_distribution<> index_dist(0, N - 1);
            int start_idx = index_dist(generator);
            
            int min_len = std::min(N / 20, 10); 
            int max_len = std::min(N / 5, 100);  
            if (max_len < min_len) max_len = min_len + 1;
            
            std::uniform_int_distribution<> len_dist(min_len, max_len);
            int segment_length = len_dist(generator);
            int end_idx = (start_idx + segment_length) % N;
            
            if (start_idx > end_idx) swap(start_idx, end_idx);

            std::reverse(current_tour.begin() + start_idx, current_tour.begin() + end_idx);
        }
        
        // --- CHAINED LOCAL SEARCH (Organic Flow) ---
        
        // 1. Run UNCONSTRAINED 2-Opt until local optimum (O(N^2)). 
        // This guarantees a strong initial untangling.
        current_tour = twoOpt(current_tour, start_time);
        
        // 2. Run CONSTRAINED Relocate-Opt (2.5-Opt) until local optimum (O(N*K)).
        // This is a fast, deep search to refine the 2-opt solution.
        vector<int> local_optimized_tour = relocateOpt(current_tour, start_time);
        
        // -----------------------------

        double local_cost = calculateTourCost(local_optimized_tour); 

        // Check for global improvement
        if (local_cost < best_cost) {
            best_cost = local_cost;
            best_tour = local_optimized_tour;
            cerr << "--- New Best Cost (" << iteration_count << "): " << fixed << best_cost << endl;
        }
    }
    // -------------------------------------------------
    

    // 3. FINAL TIME, COST, AND MOVE COUNT OUTPUT

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double elapsed_seconds = duration.count();
    
    cout << "Cost: " << std::fixed << best_cost << endl;
    cout << "Cycles: " << std::scientific << std::setprecision(1) << (double)total_moves_checked << endl;
    
    cout << "Nodes List:";
    for (int node_index : best_tour) {
        cout << " " << node_index + 1; 
    }
    cout << " " << best_tour.front() + 1 << endl; 

    cerr << "--- Optimization Complete. Final Cost: " << fixed << best_cost << endl;
    cerr << "Total Time: " << fixed << elapsed_seconds << " seconds over " << iteration_count << " starts." << endl;

    outputSolutionToFile(best_tour, "solutions_915552766.txt");

    return 0;
}
