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
#include <limits>

using namespace std;

// --- GLOBAL CONSTANTS (TUNED FOR 60 SECONDS) ---
const double MAX_RUNTIME_SECONDS = 60.0;
const double GLOBAL_TIME_LIMIT_SAFETY = MAX_RUNTIME_SECONDS - 0.5; // Stop at 58.0 seconds
// The number of nearest neighbors to consider for pruning. Higher K means better quality but slower O(N*K) searches.
const int K_NN_LIST = 75; 

// --- Global Data Structures
vector<vector<double>> D; // Distance matrix
int N; // Number of nodes
// Stores the K_NN_LIST closest neighbor indices for each node (used for pruning).
vector<vector<int>> nn_list; 
// Map to quickly find the index of a node ID in the tour vector (ID -> Index).
vector<int> node_to_index; 

// ----------------------------------------------------
// GLOBAL COUNTER AND RANDOM GENERATOR
// ----------------------------------------------------
static long long total_moves_checked = 0; 
//static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count()); 
static std::mt19937 generator(42); 
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

/**
 * @brief Calculates the total cost of a full tour.
 * @param tour The sequence of node indices (0-based).
 * @return double The total cost of the closed cycle.
 */
double calculateTourCost(const vector<int>& tour) {
    double cost = 0.0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        cost += D[tour[i]][tour[i+1]];
    }
    // Add return edge to close the cycle
    cost += D[tour.back()][tour.front()]; 
    return cost;
}

/**
 * @brief Constructs an initial tour using the Nearest Neighbor heuristic.
 * @return vector<int> The initial tour (sequence of node indices).
 */
vector<int> nearestNeighborTour() {
    vector<int> tour;
    vector<bool> visited(N, false);
    
    // Start at Node 0 (an arbitrary starting point)
    int current_node = 0;
    visited[current_node] = true;
    tour.push_back(current_node);

    // Loop N-1 times to visit the remaining nodes
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
 * @brief Reads the graph data from the input file into the distance matrix D.
 */
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
    
    // Skip remaining header lines
    std::string header_line;
    std::getline(file, header_line); 
    std::getline(file, header_line); 
    
    D.assign(N, std::vector<double>(N, 0.0)); 

    int u, v;
    double dist;

    // Read edges
    while (file >> u >> v >> dist) {
        // Adjust to 0-based indexing
        u--; 
        v--;
        
        if (u >= 0 && u < N && v >= 0 && v < N && u != v) {
            D[u][v] = dist;
            D[v][u] = dist; // Assume symmetric matrix
        }
    }

    file.close();
    cerr << "DEBUG: Graph Read Complete. N = " << N << endl;
}


// ----------------------------------------------------
// NN PRUNING LOGIC
// ----------------------------------------------------

/**
 * @brief Precomputes the K_NN_LIST closest neighbors for every node.
 */
void precomputeNNList() {
    nn_list.assign(N, vector<int>(K_NN_LIST));
    node_to_index.assign(N, 0); // Initialize map size
    for (int i = 0; i < N; ++i) {
        // Create a list of all nodes and their distance to node i
        vector<pair<double, int>> distances;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                distances.push_back({D[i][j], j});
            }
        }
        
        // Sort by distance (cost)
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
// 1. O(N^2) UNCONSTRAINED 2-OPT SWAP (The Global Sweep)
// ----------------------------------------------------

/**
 * @brief Performs a First-Improvement UNCONSTRAINED 2-Opt local search (O(N^2) per pass).
 * This is used for global untangling after a perturbation.
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

        // Safety Time check
        std::chrono::duration<double> total_duration = std::chrono::steady_clock::now() - start_time;
        if (total_duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
            return tour;
        }
        
        // O(N^2) - Iterate over all pairs of edges (i, i1) and (k, k1)
        for (int i = 0; i < N_nodes; ++i) {
            int i1 = (i + 1) % N_nodes; // Index after i
            
            // Start k at i + 2 to avoid adjacent edges, but k will wrap around due to % N_nodes
            for (int k = 0; k < N_nodes; ++k) { 
                
                int k1 = (k + 1) % N_nodes; // Index after k
                
                // Ensure the two edges are not adjacent or identical.
                if (k == i || k == i1 || k1 == i || k1 == i1) continue; 
                
                total_moves_checked++; 

                // Edge removal: (u, v) and (x, y)
                int u = tour[i];
                int v = tour[i1];
                int x = tour[k];
                int y = tour[k1]; 

                double removed_cost = D[u][v] + D[x][y];
                // Edges added: (u, x) and (v, y)
                double added_cost = D[u][x] + D[v][y];

                double gain = removed_cost - added_cost;

                if (gain > 1e-9) { 
                    // Found a profitable 2-Opt move. Execute the segment reversal.

                    // Determine the start and end of the segment to reverse
                    int start_rev = (i + 1) % N_nodes;
                    int end_rev = k;

                    if (start_rev == end_rev) {
                        // This should be avoided by the adjacency check, but good to be safe.
                        continue; 
                    }
                    
                    // Simple segment reversal (i+1 to k)
                    if (start_rev < end_rev) {
                        std::reverse(tour.begin() + start_rev, tour.begin() + end_rev + 1);
                    } else {
                        // Complex wrapping segment reversal (i+1 -> N-1, and 0 -> k)
                        // A non-wrapping swap method is safer and faster for First-Improvement
                        // This simpler approach works by only swapping if i1 < k, so we ensure that's covered.
                        // A more robust way to handle wrap is to reorder the tour sections, but for first-improvement
                        // the gain is so large it's better to just ensure the indices are in order.
                        // Since we iterate over all i and k, we rely on finding the swap where i1 < k.
                        // For the sake of robust code, we stick to the simple reverse between the two points.
                        
                        // NOTE: The previous, more complex wrapping logic is necessary for full O(N^2)
                        // if you want to avoid 'if' conditions. We'll use the user's robust logic from the prompt.
                         vector<int> temp_segment;
                         // Segment 1: [i+1, N-1]
                         temp_segment.insert(temp_segment.end(), tour.begin() + i1, tour.end());
                         // Segment 2: [0, k]
                         temp_segment.insert(temp_segment.end(), tour.begin(), tour.begin() + k + 1);

                         std::reverse(temp_segment.begin(), temp_segment.end());

                         // Reassemble the tour
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
// 2. O(N*K) CONSTRAINED RELOCATE-OPT (The Fast Refinement)
// ----------------------------------------------------
/**
 * @brief Performs a First-Improvement Relocate-Opt (1-Node Or-Opt) local search. 
 * The reinsertion point is constrained by the NN list of the predecessor node (O(N*K)).
 * @param tour The initial tour to improve.
 * @param start_time The global starting time point for time limit enforcement.
 * @return The locally optimized tour.
 */
vector<int> relocateOpt(vector<int> tour, const std::chrono::steady_clock::time_point& start_time) {
    bool improved = true;
    int N_nodes = tour.size();
    
    // Lambda to quickly update the node_to_index map (ID -> Index)
    auto update_node_to_index = [&]() {
        for(int idx = 0; idx < N_nodes; ++idx) {
            node_to_index[tour[idx]] = idx;
        }
    };
    update_node_to_index(); // Initialize map

    while (improved) {
        improved = false; 
        bool move_was_made = false;

        // Time check
        auto time_check = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = time_check - start_time;
        if (duration.count() >= GLOBAL_TIME_LIMIT_SAFETY) {
            return tour;
        }
        
        // Iterate over every node i (the node to be relocated). (N iterations)
        for (int i = 0; i < N_nodes; ++i) {
            
            // Nodes involved in the segment removal:
            int u = tour[(i - 1 + N_nodes) % N_nodes]; // Node before i (u)
            int v = tour[i];  //Node i (v)
            int w = tour[(i + 1) % N_nodes]; // Node after i (w)
            
            // Removed edges: (u, v) and (v, w)
            double removed_cost = D[u][v] + D[v][w];
            
            // Iterate over the K Nearest Neighbors of node 'u'
            // The NN node 'p_node' will be the node *before* the insertion point. (K iterations)
            for (int p_node : nn_list[u]) { 
                
                int p_idx = node_to_index[p_node]; // Index of p_node (insertion predecessor)
                
                // Nodes involved in the reinsertion:
                int p = tour[p_idx]; // Insertion predecessor (p)
                int q = tour[(p_idx + 1) % N_nodes]; // Insertion successor (q)
                
                // Skip if insertion is adjacent to removal, which leads to identity moves or complex wraps
                if (p == v || q == v || p == u || q == u) continue;
                
                total_moves_checked++; 
                
                // Edges broken by insertion: (u, w) and (p, q)
                double old_edges_insert = D[u][w] + D[p][q]; 
                
                // Edges created by move: (u, w) is replaced by (u,w), but also (u,w) is not in the tour anymore
                // The cost calculation needs to be:
                // Gain = (Removed edges) - (Added edges)
                // Removed: D[u][v] + D[v][w] + D[p][q]
                // Added: D[u][w] + D[p][v] + D[v][q]
                
                double gain = (D[u][v] + D[v][w] + D[p][q]) - (D[u][w] + D[p][v] + D[v][q]);

                if (gain > 1e-9) { 
                    // *** FIRST-IMPROVEMENT: Execute the move immediately ***
                    
                    // 1. Remove node v from its current position
                    int current_i = node_to_index[v]; 
                    tour.erase(tour.begin() + current_i);
                    
                    // 2. Calculate the new insertion index.
                    // If current_i < p_idx, erasing 'i' shifts all indices from current_i+1 to p_idx down by 1.
                    // If current_i > p_idx, erasing 'i' does not affect indices before p_idx.
                    int new_p_idx = node_to_index[p];
                    if (current_i < new_p_idx) {
                        new_p_idx -= 1;
                    }
                    
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


// ----------------------------------------------------
// MAIN EXECUTION LOGIC
// ----------------------------------------------------

int main(int argc, char* argv[]) {
    
    auto start_time = std::chrono::steady_clock::now(); 
    
    // Read file 
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    
    // 1. Read Graph Data
    readGraph(argv[1]);

    // 2. Precompute Nearest Neighbor List 
    precomputeNNList();

    // 3. Initial Best Tour (from Nearest Neighbor)
    vector<int> best_tour = nearestNeighborTour();
    double best_cost = calculateTourCost(best_tour);
    cerr << "--- Initial Cost (NN Start): " << fixed << setprecision(2) << best_cost << endl;

    int iteration_count = 0;

    // -------------------------------------------------
    // ITERATED LOCAL SEARCH (ILS) MAIN LOOP 
    // -------------------------------------------------
    while (std::chrono::steady_clock::now() - start_time < std::chrono::duration<double>(GLOBAL_TIME_LIMIT_SAFETY)) {
        
        iteration_count++;
        
        // Start from the current global best tour
        vector<int> current_tour = best_tour; 
        
        // --- 4. PERTURBATION STRATEGY (Random Segment Reversal) ---
        if (iteration_count > 1) {
            
            // Select two random, non-adjacent cut points for a random 2-Opt style move.
            std::uniform_int_distribution<> p_dist(0, N - 1);
            int p1_idx, p2_idx;
            
            // Loop until p1 and p2 are not adjacent
            do {
                p1_idx = p_dist(generator);
                p2_idx = p_dist(generator);
                // Ensure they are not the same node, and not immediate neighbors
            } while (p1_idx == p2_idx || (p1_idx + 1) % N == p2_idx || (p2_idx + 1) % N == p1_idx);
            
            // Ensure p1_idx is the smaller index for the std::reverse call
            if (p1_idx > p2_idx) std::swap(p1_idx, p2_idx);

            // Execute the random 2-Opt move (segment reversal) to "kick" the solution out.
            std::reverse(current_tour.begin() + p1_idx, current_tour.begin() + p2_idx + 1);
        }
        
        // --- 5. CHAINED LOCAL SEARCH (Optimized for Quality) ---
        
        // A. Full O(N^2) Sweep: Critical to globally untangle large crossings after perturbation.
        current_tour = twoOpt(current_tour, start_time);
        
        // B. Constrained O(N*K) Refinement: Fast, deep search to refine the already untangled tour.
        vector<int> local_optimized_tour = relocateOpt(current_tour, start_time);
        
        // -----------------------------

        double local_cost = calculateTourCost(local_optimized_tour); 

        // Check for global improvement (Greedy Acceptance)
        if (local_cost < best_cost) {
            best_cost = local_cost;
            best_tour = local_optimized_tour;
            cerr << "--- New Best Cost (Start #" << iteration_count << "): " << fixed << setprecision(2) << best_cost << endl;
        }
    }
    // -------------------------------------------------
    

    // 6. FINAL OUTPUT
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    cout << "Cost: " << std::fixed << setprecision(2) << best_cost << endl;
    cout << "Cycles: " << std::scientific << std::setprecision(1) << (double)total_moves_checked << endl;
    
    // Output nodes in 1-based indexing
    cout << "Nodes List:";
    for (int node_index : best_tour) {
        cout << " " << node_index + 1; 
    }
    cout << " " << best_tour.front() + 1 << endl; 

    cerr << "Total Time: " << fixed << setprecision(3) << duration.count() << " seconds over " << iteration_count << " starts." << endl;

    // Output to file (using a standard filename)
    outputSolutionToFile(best_tour, "final_solution12.txt");

    return 0;
}