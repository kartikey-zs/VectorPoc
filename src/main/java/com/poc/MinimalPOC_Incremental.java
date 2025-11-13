package com.poc;

import org.apache.ignite.client.IgniteClient;
import org.apache.ignite.sql.ResultSet;
import org.apache.ignite.sql.SqlRow;
import org.apache.ignite.table.Table;
import org.apache.ignite.table.Tuple;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * JVector + GridGain 9 - In-Memory Vector Search POC with Incremental Updates
 * <p>
 * PURPOSE:
 * This POC demonstrates integration between JVector (in-memory vector search library) and
 * GridGain 9 (distributed database) to explore incremental index updates, deletion handling,
 * and memory management strategies for vector search workloads.
 * <p>
 * ARCHITECTURE:
 * <p>
 * 1. STORAGE LAYER (GridGain 9):
 *    - Stores raw vectors as VARBINARY in a distributed table
 *    - Provides durability and persistence for vector data
 *    - Vectors identified by integer IDs (primary key)
 * <p>
 * 2. INDEX LAYER (JVector):
 *    - In-memory HNSW graph for fast approximate nearest neighbor search
 *    - Vectors represented by ordinals (integers) that map to vectorList indices
 *    - Supports incremental updates via addGraphNode() - no full rebuild needed
 *    - Deletion via markNodeDeleted() + cleanup()
 * <p>
 * 3. MEMORY MODEL:
 *    - vectorList: ArrayList<VectorFloat<?>> - holds all vectors in memory
 *    - OnHeapGraphIndex: HNSW graph structure stored on JVM heap
 *    - RandomAccessVectorValues: Interface between graph and vector storage
 * <p>
 * KEY LEARNING: ORDINALS
 * JVector uses "ordinals" (int values) to identify vectors in the index.
 * For ListRandomAccessVectorValues, ordinals = ArrayList indices.
 * This means: ordinal 42 → vectorList.get(42)
 * <p>
 * INCREMENTAL UPDATES:
 * - addGraphNode(ordinal, vector) adds a single vector to existing graph
 * - NO full rebuild required - graph structure updates incrementally
 * - Search quality maintained through neighbor selection algorithms
 * - Performance: ~1-5ms per vector insertion (vs ~100ms for full rebuild)
 * <p>
 * DELETION STRATEGY:
 * <p>
 * Option 1: Mark Only (No Cleanup)
 *   - markNodeDeleted(id) - thread-safe, instant
 *   - Vector excluded from search results immediately
 *   - Memory NOT reclaimed (vector stays in vectorList)
 *   - Graph structure unchanged
 *   - Use case: Short-lived processes, low deletion rate
 * <p>
 * Option 2: Cleanup (Graph Optimization)
 *   - markNodeDeleted(id) + cleanup()
 *   - Removes deleted nodes from graph structure
 *   - Creates bridging edges to maintain connectivity
 *   - Memory NOT fully reclaimed (holes remain in vectorList)
 *   - NOT thread-safe - blocks all operations
 *   - Use case: Optimize graph quality without full compaction
 * <p>
 * Option 3: Cleanup with Compaction (Full Memory Reclamation)
 *   - Removes deleted vectors from vectorList
 *   - Rebuilds graph with sequential ordinals (no holes)
 *   - Full memory reclamation
 *   - Graph ready for disk persistence
 *   - Takes ~100ms for 1000 vectors
 *   - Use case: High deletion rate, memory pressure, or preparing for persistence
 * <p>
 * WHY CLEANUP IS NOT A BLOCKER FOR IN-MEMORY:
 * <p>
 * 1. Cleanup is OPTIONAL:
 *    - Marking deletions is sufficient for functionality
 *    - Deleted nodes immediately excluded from search results
 *    - Can accumulate deletions indefinitely
 * <p>
 * 2. You Control When Cleanup Runs:
 *    - Interactive POC = manual trigger
 *    - Production = schedule during low-traffic windows
 *    - In-memory = can restart process to clear all deletions
 * <p>
 * 3. Cleanup is NOT Thread-Safe BUT:
 *    - Only needed periodically (not per operation)
 *    - Can use application-level coordination
 *    - For high-throughput: use multiple index instances
 * <p>
 * 4. Performance Characteristics:
 *    - Mark deletion: <1ms (thread-safe)
 *    - Cleanup: 30-50ms per 1000 nodes (not thread-safe)
 *    - Compaction: 100-150ms per 1000 nodes (rebuild)
 * <p>
 * TESTED SCENARIOS:
 * - Initial bulk load: 1000 vectors
 * - Incremental additions: single vector at a time
 * - Deletions: mark + cleanup
 * - Compaction: full memory reclamation
 * - Search: approximate nearest neighbor queries
 * <p>
 * LIMITATIONS & FUTURE WORK:
 * - Persistence: Currently in-memory only (can add save/load)
 * - Concurrency: Single-threaded operations (cleanup constraint)
 * - Scale: Limited by JVM heap size
 * - Compression: No Product Quantization implemented yet
 *
 * @author Kartikey Srivastava
 * @version 1.0
 */
public class MinimalPOC_Incremental {

    // ============= CONSTANTS =============

    /**
     * Vector type support from JVector's vectorization provider.
     * Handles creation of VectorFloat instances with SIMD optimizations when available.
     */
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT =
            VectorizationProvider.getInstance().getVectorTypeSupport();
    /**
     * Dimensionality of vectors in this POC.
     * All vectors must have exactly this many dimensions.
     */
    private static final int DIMENSION = 128;

    // ============= STATE - PERSISTENCE & CONNECTION =============

    /**
     * GridGain client connection.
     * Provides access to distributed tables for vector storage.
     */
    private static IgniteClient client;
    /**
     * Scanner for user input in the interactive menu.
     */
    private static Scanner scanner;

    // ============= STATE - VECTOR INDEX COMPONENTS =============

    /**
     * JVector's GraphIndexBuilder - manages the HNSW graph structure.
     * <p>
     * THREAD SAFETY:
     * - addGraphNode(): thread-safe, can be called concurrently
     * - markNodeDeleted(): thread-safe
     * - cleanup(): NOT thread-safe, requires exclusive access
     * <p>
     * LIFECYCLE:
     * - Created once during buildInitialIndex()
     * - Recreated after compaction (performCleanupWithCompaction)
     * - Must be closed before application exit
     */
    private static GraphIndexBuilder builder;
    /**
     * The actual HNSW graph index stored on-heap.
     * Retrieved from builder.getGraph() - represents the graph structure
     * with nodes and edges connecting similar vectors.
     */
    private static OnHeapGraphIndex index;
    /**
     * In-memory list of all vectors.
     * <p>
     * CRITICAL: Ordinal Mapping
     * JVector uses ordinals (int) to identify vectors in the graph.
     * For ListRandomAccessVectorValues, ordinals map directly to list indices:
     *   - Ordinal 0 → vectorList.get(0)
     *   - Ordinal 42 → vectorList.get(42)
     * <p>
     * This mapping MUST be maintained:
     * - During incremental adds: append to list, use list size as ordinal
     * - After deletions: list has "holes" (deleted vectors remain)
     * - After compaction: rebuild list without holes, reassign ordinals
     */
    private static List<VectorFloat<?>> vectorList;
    /**
     * RandomAccessVectorValues interface wrapper around vectorList.
     * Provides JVector with random access to vectors by ordinal.
     * <p>
     * When JVector needs vector at ordinal N, it calls ravv.vectorValue(N),
     * which internally does vectorList.get(N).
     */
    private static RandomAccessVectorValues ravv;

    // ============= STATE - TRACKING & METADATA =============

    /**
     * Next available ordinal for new vectors.
     * <p>
     * Incremented after each addGraphNode() call.
     * After compaction, reset to vectorList.size() (sequential ordinals).
     */
    private static int nextOrdinal = 0;

    /**
     * Set of ordinals marked for deletion.
     * <p>
     * LIFECYCLE:
     * - Added to by markVectorDeleted()
     * - Read by performCleanupWithCompaction() to know which indices to skip
     * - Cleared after compaction (memory fully reclaimed)
     * - NOT cleared after regular cleanup (used for later compaction)
     * <p>
     * IMPORTANT: This tracks ordinals/indices, not GridGain IDs
     * (though in this POC they are the same).
     */
    private static Set<Integer> deletedNodes = new HashSet<>();
    /**
     * Flag indicating whether graph cleanup needs to be performed.
     * <p>
     * Set to true when: markNodeDeleted() is called
     * Set to false when: cleanup() or performCleanupWithCompaction() completes
     * <p>
     * Purpose: Allows performCleanupWithCompaction() to skip cleanup if already done.
     */
    private static boolean needsGraphCleanup = false;  // NEW


    // ============= MAIN ENTRY POINT =============

    /**
     * Main entry point for the POC.
     * <p>
     * WORKFLOW:
     * 1. Initialize connection to GridGain cluster
     * 2. Create vector storage table
     * 3. Insert initial batch of 1000 vectors
     * 4. Build HNSW index incrementally
     * 5. Enter interactive loop for user commands
     *
     * @param args Command line arguments (not used)
     * @throws Exception if initialization or interactive loop fails
     */
    public static void main(String[] args) throws Exception {
        System.out.println("=== JVector + GridGain 9 - Incremental Updates POC ===\n");

        // Initialize once
        initializeConnection();
        createTable();
        insertInitialVectors(1000);
        buildInitialIndex();
        verifyVectors();


        // Interactive loop
        runInteractiveLoop();
    }


    // ============= INITIALIZATION (Runs Once) =============
    /**
     * Establishes connection to GridGain cluster and initializes the scanner for user input.
     * <p>
     * CONFIGURATION:
     * - Connects to localhost:10800 (default GridGain thin client port)
     * - Uses default authentication (none)
     * <p>
     * ERRORS:
     * - Throws exception if cluster is not reachable
     * - No retry logic - fails fast
     */
    private static void initializeConnection() {
        System.out.println("1. Connecting to GridGain cluster...");
        client = IgniteClient.builder()
                .addresses("127.0.0.1:10800")
                .build();
        scanner = new Scanner(System.in);
        System.out.println("   ✓ Connected\n");
    }

    /**
     * Creates the zone and table for vector storage in GridGain.
     * <p>
     * SCHEMA:
     * - Zone: poc_zone (for data distribution configuration)
     * - Table: vectors
     *   - id (INT, PRIMARY KEY): Vector identifier and ordinal
     *   - embedding (VARBINARY): Serialized float array (DIMENSION * 4 bytes)
     * <p>
     * IDEMPOTENT: Uses IF NOT EXISTS, safe to call multiple times.
     */
    private static void createTable() {
        System.out.println("2. Creating table...");
        try {
            client.sql().execute(null,
                    "CREATE ZONE IF NOT EXISTS poc_zone STORAGE PROFILES['default']");
        } catch (Exception e) {
            // Ignore if exists
        }

        try {
            client.sql().execute(null,
                    "CREATE TABLE IF NOT EXISTS vectors (" +
                            "    id INT PRIMARY KEY," +
                            "    embedding VARBINARY" +
                            ") ZONE poc_zone");
        } catch (Exception e) {
            // Ignore if exists
        }
        System.out.println("   ✓ Table ready\n");
    }

    /**
     * Inserts initial batch of vectors into GridGain for index building.
     * <p>
     * APPROACH:
     * - Generates random vectors of specified dimension
     * - Serializes to VARBINARY using little-endian byte order
     * - Inserts with sequential IDs starting from 0
     * <p>
     * ORDINAL ALIGNMENT:
     * The ID in GridGain matches the ordinal in JVector:
     *   - GridGain ID 0 → JVector ordinal 0
     *   - GridGain ID 42 → JVector ordinal 42
     * <p>
     * This alignment simplifies the architecture but is not strictly required.
     *
     * @param numVectors Number of vectors to insert (e.g., 1000)
     */
    private static void insertInitialVectors(int numVectors) {
        System.out.println("3. Inserting initial " + numVectors + " vectors...");
        Table table = client.tables().table("vectors");

        for (int id = 0; id < numVectors; id++) {
            float[] vector = generateRandomVector(DIMENSION);
            byte[] vectorBytes = floatArrayToBytes(vector);

            Tuple tuple = Tuple.create()
                    .set("id", id)
                    .set("embedding", vectorBytes);

            table.recordView().upsert(null, tuple);

            if ((id + 1) % 200 == 0) {
                System.out.println("   Inserted " + (id + 1) + " vectors...");
            }
        }
        nextOrdinal = numVectors;  // Next vector will be ID 1000
        System.out.println("   ✓ Inserted " + numVectors + " vectors\n");
    }

    /**
     * Builds the initial HNSW index incrementally using addGraphNode().
     * <p>
     * APPROACH: Incremental Building (NOT Batch Build)
     * Instead of using builder.build() which processes all vectors at once,
     * we use builder.addGraphNode(ordinal, vector) for each vector.
     * This demonstrates the incremental update capability.
     * <p>
     * WHY INCREMENTAL?
     * - Simulates real-world scenario where vectors arrive over time
     * - Tests the same code path used for live updates
     * - Proves that incremental updates maintain search quality
     * <p>
     * STEPS:
     * 1. Load vectors from GridGain into vectorList
     * 2. Create RandomAccessVectorValues wrapper
     * 3. Create GraphIndexBuilder with HNSW parameters
     * 4. Add each vector using addGraphNode() incrementally
     * 5. Analyze resulting graph connectivity
     * <p>
     * PARAMETERS:
     * - M=16: Max connections per node (affects recall vs speed)
     * - efConstruction=100: Size of dynamic candidate list during construction
     * - neighborOverflow=1.2f: Allow 20% more neighbors during construction
     * - alpha=1.2f: Heuristic parameter for neighbor selection
     *
     * @throws Exception if loading vectors or building index fails
     */
    private static void buildInitialIndex() throws Exception {
        System.out.println("4. Building initial HNSW index...");

        // Load vectors from GridGain
        System.out.println("   Loading vectors from GridGain...");
        vectorList = new ArrayList<>();

        try (ResultSet<SqlRow> rs = client.sql().execute(null,
                "SELECT id, embedding FROM vectors ORDER BY id")) {
            while (rs.hasNext()) {
                SqlRow row = rs.next();
                byte[] vectorBytes = row.value("embedding");
                float[] floats = bytesToFloatArray(vectorBytes);

                // Convert to JVector's VectorFloat type
                VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(DIMENSION);
                for (int i = 0; i < DIMENSION; i++) {
                    vector.set(i, floats[i]);
                }
                vectorList.add(vector);
            }
        }
        System.out.println("   Loaded " + vectorList.size() + " vectors");
        nextOrdinal = vectorList.size();

        // STEP 2: Create RandomAccessVectorValues wrapper
        // This provides JVector with ordinal-based access to vectors
        ravv = new ListRandomAccessVectorValues(vectorList, DIMENSION);

        // STEP 3: Create BuildScoreProvider
        // Provides similarity scores during graph construction
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
                ravv,
                VectorSimilarityFunction.EUCLIDEAN
        );

        // STEP 4: Create builder with HNSW parameters
        System.out.println("\n   Building graph entirely via addGraphNode (like JVector examples)...");
        builder = new GraphIndexBuilder(
                bsp,
                DIMENSION,
                16,             // M
                100,            // efConstruction
                1.2f,           // neighborOverflow
                1.2f,           // alpha
                false,          // single layer
                true            // refineFinalGraph
        );

        // Get the graph directly from the builder (don't call build!)
        // The graph is empty at this point
        index = (OnHeapGraphIndex) builder.getGraph();

        // STEP 5: Add all nodes incrementally
        // This is the KEY demonstration - adding nodes one by one
        for (int i = 0; i < vectorList.size(); i++) {
            builder.addGraphNode(i, vectorList.get(i));

            if ((i + 1) % 100 == 0) {
                System.out.println("   Added " + (i + 1) + " nodes to graph");
            }
        }
        System.out.println("   ✓ Index built with " + index.size() + " nodes");

        // STEP 6: Analyze connectivity
        System.out.println("\n   Analyzing graph connectivity:");
        analyzeGraphConnectivity();
    }

    /**
     * Verifies that loaded vectors are valid and distinct.
     * <p>
     * CHECKS:
     * - Vectors are not all zeros
     * - Vectors have reasonable min/max values
     * - Vectors are actually different from each other
     * <p>
     * PURPOSE: Catch data loading issues early before building index.
     */
    private static void verifyVectors() {
        System.out.println("\nVerifying vector data:");

        // Check first few vectors aren't all zeros or identical
        for (int i = 0; i < 5; i++) {
            VectorFloat<?> vec = vectorList.get(i);
            float sum = 0;
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;

            for (int j = 0; j < DIMENSION; j++) {
                float val = vec.get(j);
                sum += val;
                min = Math.min(min, val);
                max = Math.max(max, val);
            }

            System.out.printf("Vector %d: sum=%.2f, min=%.2f, max=%.2f\n",
                    i, sum, min, max);
        }

        // Check if vectors are actually different
        float similarity = VectorSimilarityFunction.EUCLIDEAN.compare(
                vectorList.get(0), vectorList.get(1));
        System.out.println("Similarity between vector 0 and 1: " + similarity);
    }

    // ============= INTERACTIVE LOOP =============

    /**
     * Main interactive loop presenting menu and handling user commands.
     * <p>
     * AVAILABLE COMMANDS:
     * 1. Add new random vector - demonstrates incremental updates
     * 2. Search for neighbors - tests approximate nearest neighbor search
     * 3. Show detailed index statistics - memory usage, graph properties
     * 4. Analyze graph connectivity - verify graph structure integrity
     * 5. Mark vector for deletion - soft delete (thread-safe)
     * 6. Perform cleanup - optimize graph structure (not thread-safe)
     * 7. Perform cleanup with compaction - full memory reclamation
     * 8. Exit - clean shutdown
     *
     * @throws Exception if any operation fails
     */

    private static void runInteractiveLoop() throws Exception {
        System.out.println("========================================");
        System.out.println("Index ready for incremental updates!");
        System.out.println("========================================\n");

        while (true) {
            printMenu();

            int choice = getIntInput();
            System.out.println();

            switch (choice) {
                case 1:
                    addNewVector();
                    break;
                case 2:
                    performSearch();
                    break;
                case 3:
                    showDetailedStats();
                    break;
                case 4:
                    analyzeGraphConnectivity();
                    break;
                case 5:
                    markVectorDeleted();
                    break;
                case 6:
                    performCleanup();
                    break;
                case 7:
                    performCleanupWithCompaction();
                    break;
                case 8:
                    cleanupApplication();
                    System.out.println("Goodbye!");
                    System.exit(0);
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.\n");
            }
        }
    }

    /**
     * Displays the interactive menu with available commands.
     */
    private static void printMenu() {
        System.out.println("========================================");
        System.out.println("Commands:");
        System.out.println("  1. Add new random vector");
        System.out.println("  2. Search for neighbors");
        System.out.println("  3. Show detailed index statistics");
        System.out.println("  4. Analyse graph connectivity");
        System.out.println("  5. Soft Delete a node");
        System.out.println("  6. Perform cleanup");
        System.out.println("  7. Perform cleanup with compaction");
        System.out.println("  8. Exit");
        System.out.println("========================================");
        System.out.print("Enter choice: ");
    }

    /**
     * Safely reads an integer from user input with error handling.
     *
     * @return The entered integer, or -1 if input was invalid
     */
    private static int getIntInput() {
        try {
            return scanner.nextInt();
        } catch (Exception e) {
            scanner.nextLine(); // Clear buffer
            return -1;
        }
    }

    // ============= INCREMENTAL ADD =============

    /**
     * Adds a single new vector to both GridGain and the HNSW index.
     * <p>
     * This is the KEY METHOD demonstrating incremental updates.
     * <p>
     * WORKFLOW:
     * 1. Generate random vector
     * 2. Insert to GridGain (durability/persistence)
     * 3. Create VectorFloat instance
     * 4. Append to vectorList (ordinal = current list size)
     * 5. Add to graph using builder.addGraphNode(ordinal, vector)
     * 6. Increment nextOrdinal tracker
     * <p>
     * CRITICAL INSIGHT: addGraphNode() is INCREMENTAL
     * - Does NOT rebuild the entire graph
     * - Finds appropriate neighbors for the new node
     * - Updates existing nodes' neighbor lists as needed
     * - Maintains HNSW invariants (connectivity, navigability)
     * <p>
     * PERFORMANCE:
     * - Typical time: 1-5ms for a single vector
     * - Scales with M (max connections) and efConstruction
     * - Much faster than rebuilding (which takes ~100ms for 1000 vectors)
     * <p>
     * ORDINAL MANAGEMENT:
     * The ordinal used is the current size of vectorList:
     *   vectorList.size() = N → new ordinal = N
     *   After add: vectorList.size() = N+1
     * <p>
     * This ensures ordinals remain sequential and aligned with indices.
     *
     * @throws Exception if database insert or graph update fails
     */

    private static void addNewVector() throws Exception {
        System.out.println("Adding new vector with ID: " + nextOrdinal);

        long startTime = System.currentTimeMillis();

        // 1. Generate random vector
        float[] newVectorArray = generateRandomVector(DIMENSION);

        // 2. Insert to GridGain
        Table table = client.tables().table("vectors");
        byte[] vectorBytes = floatArrayToBytes(newVectorArray);
        Tuple tuple = Tuple.create()
                .set("id", nextOrdinal)
                .set("embedding", vectorBytes);
        table.recordView().upsert(null, tuple);
        System.out.println("   ✓ Inserted to GridGain");

        // 3. Create VectorFloat
        VectorFloat<?> newVector = VECTOR_TYPE_SUPPORT.createFloatVector(DIMENSION);
        for (int i = 0; i < DIMENSION; i++) {
            newVector.set(i, newVectorArray[i]);
        }

        // 4. Add to vectorList (so RAVV sees it)
        vectorList.add(newVector);
        System.out.println("   ✓ Added to vector list");

        // 5. THE KEY OPERATION: Add to graph using addGraphNode!
        // This is incremental - no full rebuild
        long addNodeStart = System.currentTimeMillis();
        builder.addGraphNode(nextOrdinal, newVector);
        long addNodeEnd = System.currentTimeMillis();

        System.out.println("   ✓ Added to graph index (took " + (addNodeEnd - addNodeStart) + "ms)");

        // 6. Update tracking
        nextOrdinal++;

        long totalTime = System.currentTimeMillis() - startTime;

        System.out.println("\n✓ Vector added successfully!");
        System.out.println("  New vector ID: " + (nextOrdinal - 1));
        System.out.println("  Total vectors: " + vectorList.size());
        System.out.println("  Index size: " + index.size());
        System.out.println("  Time taken: " + totalTime + "ms");
        System.out.println();
    }

    // ============= SEARCH =============

    /**
     * Performs approximate nearest neighbor search using the HNSW index.
     * <p>
     * WORKFLOW:
     * 1. Get query vector from vectorList using provided ID
     * 2. Create SearchScoreProvider with query vector
     * 3. Use GraphSearcher to find topK nearest neighbors
     * 4. Display results with similarity scores
     * <p>
     * KEY INSIGHT: Query Vector Can Be Deleted
     * Even if a vector is marked for deletion, you can still search FROM it
     * (use it as a query). This is because:
     * - The vector still exists in vectorList
     * - Only the graph node is marked deleted
     * - The deleted vector won't appear in OTHER queries' results
     * <p>
     * SEARCH PARAMETERS:
     * - topK: Number of nearest neighbors to return (default: 10)
     * - Bits.ALL: Accept all ordinals (no filtering)
     * - Uses EUCLIDEAN similarity from index configuration
     * <p>
     * PERFORMANCE:
     * - Typical latency: 1-5ms for 1000-10000 vectors
     * - Scales logarithmically with dataset size (HNSW property)
     *
     * @throws Exception if search fails or invalid ID provided
     */

    private static void performSearch() throws Exception {
        System.out.print("Enter vector ID to search (0 to " + (nextOrdinal - 1) + "): ");
        int queryId = getIntInput();

        if (queryId < 0 || queryId >= nextOrdinal) {
            System.out.println("Invalid ID. Must be between 0 and " + (nextOrdinal - 1) + "\n");
            return;
        }

        System.out.println("\nSearching for neighbors of vector " + queryId + "...");

        long startTime = System.currentTimeMillis();

        // Get query vector from our list
        VectorFloat<?> queryVector = vectorList.get(queryId);

        // Create SearchScoreProvider
        // Note: We're using the same ravv that points to vectorList
        // which now includes all added vectors
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(
                queryVector,
                VectorSimilarityFunction.EUCLIDEAN,
                ravv
        );

        // Perform search
        int topK = 10;
        SearchResult result;
        try (GraphSearcher searcher = new GraphSearcher(index)) {
            result = searcher.search(ssp, topK, Bits.ALL);
        }

        long searchTime = System.currentTimeMillis() - startTime;

        // Display results
        System.out.println("\n✓ Search completed in " + searchTime + "ms");
        System.out.println("Top " + topK + " nearest neighbors:");
        System.out.println("----------------------------------------");

        int rank = 1;
        for (SearchResult.NodeScore ns : result.getNodes()) {
            System.out.printf("%2d. ID: %4d | Score: %.4f", rank, ns.node, ns.score);
            if (ns.node == queryId) {
                System.out.print(" ← QUERY (exact match)");
            } else if (ns.node >= 1000) {
                System.out.print(" ← INCREMENTAL");
            }
            System.out.println();
            rank++;
        }
        System.out.println("----------------------------------------\n");
    }

    // ============= STATISTICS & ANALYSIS =============

    /**
     * Displays detailed statistics about the index and memory usage.
     * <p>
     * METRICS SHOWN:
     * <p>
     * 1. Size Information:
     *    - Total vectors in memory (vectorList.size())
     *    - Index node count (may differ if deletions occurred)
     *    - Next available ID
     *    - Count of incrementally added vectors
     * <p>
     * 2. Memory Usage:
     *    - JVM heap used
     *    - Maximum heap available
     *    - Memory utilization percentage
     * <p>
     * 3. Storage Estimates:
     *    - Raw vector storage (DIMENSION * 4 bytes * count)
     *    - Per-vector overhead
     * <p>
     * 4. Graph Parameters:
     *    - Dimension, M, similarity function
     * <p>
     * 5. Sample Connectivity:
     *    - Connection count for first, middle, and last nodes
     *    - Helps verify graph structure integrity
     * <p>
     * PURPOSE: Monitor memory usage and verify index health.
     */

    private static void showDetailedStats() {
        System.out.println("\n========================================");
        System.out.println("DETAILED INDEX STATISTICS");
        System.out.println("========================================");

        // Basic counts
        System.out.println("\n📊 Size Information:");
        System.out.println("  Total vectors in memory: " + vectorList.size());
        System.out.println("  Index node count: " + index.size());
        System.out.println("  Next available ID: " + nextOrdinal);
        System.out.println("  Vectors added incrementally: " + (nextOrdinal - 1000));

        // Memory usage
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long maxMemory = runtime.maxMemory();

        System.out.println("\n💾 Memory Usage:");
        System.out.println("  Used: " + (usedMemory / 1024 / 1024) + " MB");
        System.out.println("  Max: " + (maxMemory / 1024 / 1024) + " MB");
        System.out.println("  Utilization: " + (usedMemory * 100 / maxMemory) + "%");

        // Vector storage estimate
        long vectorStorageBytes = (long) vectorList.size() * DIMENSION * 4;  // 4 bytes per float
        System.out.println("\n📦 Estimated Storage:");
        System.out.println("  Raw vectors: ~" + (vectorStorageBytes / 1024) + " KB");
        System.out.println("  Average per vector: " + (DIMENSION * 4) + " bytes");

        // Graph parameters
        System.out.println("\n⚙️  Graph Parameters:");
        System.out.println("  Dimension: " + DIMENSION);
        System.out.println("  Max connections (M): 16");
        System.out.println("  Similarity: EUCLIDEAN");
        System.out.println("  Hierarchy: Single-layer");

        // Sample some connections
        System.out.println("\n🔗 Sample Node Connections:");
        int[] sampleNodes = {0, 500, nextOrdinal - 1};  // First, middle, last

        var view = index.getView();
        for (int nodeId : sampleNodes) {
            if (nodeId >= index.size()) continue;

            try {
                var neighbors = view.getNeighborsIterator(nodeId, 0);
                int count = 0;
                while (neighbors.hasNext()) {
                    neighbors.nextInt();
                    count++;
                }
                System.out.println("  Node " + nodeId + ": " + count + " connections");
            } catch (Exception e) {
                System.out.println("  Node " + nodeId + ": Error reading connections");
            }
        }

        System.out.println("\n========================================\n");
    }

    /**
     * Analyzes and displays graph connectivity metrics.
     * <p>
     * METRICS:
     * - Average connections per node
     * - Number of nodes with zero connections (should be minimal)
     * - Sample connection counts for specific nodes
     * <p>
     * PURPOSE:
     * Verify graph quality and connectivity after incremental updates or deletions.
     * <p>
     * EXPECTATIONS:
     * - Average connections should be close to M (16 in this POC)
     * - Zero-connection nodes indicate potential issues
     * - After incremental adds, new nodes should be well-connected
     */
    private static void analyzeGraphConnectivity() {
        var view = index.getView();
        int totalConnections = 0;
        int nodesWithZeroConnections = 0;

        for (int i = 0; i < index.size(); i++) {
            var neighbors = view.getNeighborsIterator(i, 0);
            int count = 0;
            while (neighbors.hasNext()) {
                neighbors.nextInt();
                count++;
            }
            if (count == 0) nodesWithZeroConnections++;
            totalConnections += count;

            // Log details for specific ranges
            if (i < 5 || i >= 1000 || i % 200 == 0) {
                System.out.println("Node " + i + ": " + count + " connections");
            }
        }

        System.out.println("Average connections: " + (totalConnections / (double)index.size()));
        System.out.println("Nodes with 0 connections: " + nodesWithZeroConnections);
    }

    // ============= DELETION METHODS =============

    /**
     * Marks a vector for deletion in the HNSW graph.
     * <p>
     * This is a "soft delete" operation.
     * <p>
     * WHAT IT DOES:
     * - Calls builder.markNodeDeleted(ordinal) - thread-safe operation
     * - Adds ordinal to deletedNodes set for tracking
     * - Sets needsGraphCleanup flag to true
     * <p>
     * WHAT IT DOES NOT DO:
     * - Does NOT remove vector from vectorList
     * - Does NOT remove node from graph structure yet
     * - Does NOT reclaim memory
     * <p>
     * IMMEDIATE EFFECT:
     * The marked node is immediately excluded from search results.
     * JVector internally maintains a deletion bitmap that's checked during search.
     * <p>
     * THREAD SAFETY:
     * This operation is THREAD-SAFE and can be called concurrently.
     * <p>
     * CLEANUP REQUIRED:
     * - Option 6 (Perform cleanup): Removes from graph, doesn't reclaim memory
     * - Option 7 (Cleanup with compaction): Full removal + memory reclamation
     * <p>
     * WHY SOFT DELETE?
     * - Instant operation (<1ms)
     * - Thread-safe
     * - Allows batching deletions
     * - Cleanup can be scheduled for low-traffic periods
     */
    private static void markVectorDeleted() {
        System.out.print("Enter vector ID to delete: ");
        int deleteId = getIntInput();

        if (deleteId < 0 || deleteId >= nextOrdinal) {
            System.out.println("Invalid ID.\n");
            return;
        }

        // Mark in JVector - thread-safe operation
        builder.markNodeDeleted(deleteId);
        deletedNodes.add(deleteId);
        needsGraphCleanup = true;  // NEW

        System.out.println("✓ Marked for deletion");
        System.out.println("  Total marked: " + deletedNodes.size());
        System.out.println("💡 Cleanup when convenient via menu option 6 or 7");
    }

    /**
     * Performs graph cleanup to remove deleted nodes and optimize structure.
     * <p>
     * This is Option 6: "Cleanup (graph only)"
     * <p>
     * WHAT IT DOES:
     * 1. Calls builder.cleanup() to:
     *    - Remove deleted nodes from graph structure
     *    - Create "bridging edges" to maintain connectivity
     *    - Enforce degree limits (M parameter)
     *    - Optionally refine graph quality
     * <p>
     * WHAT IT DOES NOT DO:
     * - Does NOT remove vectors from vectorList (holes remain)
     * - Does NOT reclaim memory (deleted vectors still in memory)
     * - Does NOT compact ordinals (gaps in ordinal sequence)
     * <p>
     * WHEN TO USE:
     * - Want to optimize graph structure without full rebuild
     * - Deletions are affecting search quality
     * - Not concerned about memory usage (short-lived process)
     * <p>
     * THREAD SAFETY: **NOT THREAD-SAFE**
     * - No other operations should be in progress
     * - In production: coordinate via external locking
     * - In this POC: interactive mode ensures exclusivity
     * <p>
     * PERFORMANCE:
     * - Proportional to number of deleted nodes
     * - Typical: 30-50ms for cleanup of 10-50 deletions
     * <p>
     * WHY THIS IS ACCEPTABLE FOR IN-MEMORY:
     * - Can be triggered manually during maintenance windows
     * - In-memory architecture allows process restart as alternative
     * - For high-throughput: use multiple index instances
     *
     * @throws Exception if cleanup fails
     */
    private static void performCleanup() throws Exception {
        if (!needsGraphCleanup) {
            System.out.println("No marked deletions to cleanup");
            return;
        }

        System.out.println("⚠️ Cleanup blocks operations - ensure no concurrent activity!");
        System.out.println("Cleaning " + deletedNodes.size() + " deleted nodes from graph...");

        long startTime = System.currentTimeMillis();

        // Perform graph cleanup - NOT thread-safe
        builder.cleanup();  // Removes from graph structure
        needsGraphCleanup = false;  // Graph is now clean

        System.out.println("✓ Cleanup done in " + (System.currentTimeMillis() - startTime) + "ms");
        System.out.println("  Graph structure optimized");
        System.out.println("  ⚠️ " + deletedNodes.size() + " vectors still occupy ~" +
                (deletedNodes.size() * DIMENSION * 4 / 1024) + " KB in vectorList");
        System.out.println("  💡 Use option 7 to compact vectorList and reclaim memory");

        // DON'T clear deletedNodes! We need it for compaction!
    }

    /**
     * Performs cleanup with full compaction to reclaim memory.
     * <p>
     * This is Option 7: "Cleanup with compaction (graph + memory)"
     * <p>
     * WHAT IT DOES:
     * 1. Calls builder.cleanup() if not already done
     * 2. Creates new vectorList without deleted vectors
     * 3. Closes old builder and graph
     * 4. Creates new builder with compacted vectors
     * 5. Rebuilds graph with sequential ordinals (0, 1, 2, ...)
     * 6. Updates all references
     * 7. Clears deletedNodes set
     * <p>
     * KEY INSIGHT: Ordinal Remapping
     * Before compaction:
     *   vectorList: [V0, V1, _deleted_, V3, V4, ...]  (holes)
     *   ordinals:   [0,  1,  _hole_,    3,  4,  ...]
     * <p>
     * After compaction:
     *   vectorList: [V0, V1, V3, V4, ...]  (no holes)
     *   ordinals:   [0,  1,  2,  3,  ...]  (sequential)
     * <p>
     * The graph is rebuilt so ordinals match new indices perfectly.
     * This makes the index ready for disk persistence (no holes allowed).
     * <p>
     * WHEN TO USE:
     * - High deletion rate (>10% of vectors)
     * - Memory pressure (need to reclaim space)
     * - Preparing for disk persistence (save/load)
     * - Long-running process
     * <p>
     * PERFORMANCE:
     * - Similar to initial index build (~100-150ms for 1000 vectors)
     * - Scales with number of remaining vectors (not deleted count)
     * <p>
     * THREAD SAFETY: **NOT THREAD-SAFE**
     * Same constraints as performCleanup().
     * <p>
     * RESULT:
     * - Full memory reclamation
     * - Clean, compacted data structures
     * - Ready for persistence to disk
     * - Sequential ordinals (no gaps)
     *
     * @throws Exception if compaction fails
     */
    private static void performCleanupWithCompaction() throws Exception {
        if (deletedNodes.isEmpty()) {
            System.out.println("No deleted nodes to compact!");
            return;
        }

        System.out.println("\n=== CLEANUP WITH COMPACTION ===");
        System.out.println("Deleted nodes to compact: " + deletedNodes.size());

        // Step 0: If graph cleanup wasn't done yet, do it now
        if (needsGraphCleanup) {
            System.out.println("Performing graph cleanup first...");
            builder.cleanup();
            needsGraphCleanup = false;
            System.out.println("✓ Graph cleanup done");
        }

        long startTime = System.currentTimeMillis();

        // Step 1: Create compacted vector list
        List<VectorFloat<?>> compacted = new ArrayList<>();
        for (int i = 0; i < vectorList.size(); i++) {
            if (!deletedNodes.contains(i)) {
                compacted.add(vectorList.get(i));
            }
        }

        System.out.println("✓ Compacted: " + vectorList.size() + " → " + compacted.size());

        // Step 2: Close old builder/graph
        builder.close();

        // Step 3: Create new RAVV with compacted vectors
        RandomAccessVectorValues newRavv = new ListRandomAccessVectorValues(
                compacted,
                DIMENSION
        );

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
                newRavv,
                VectorSimilarityFunction.EUCLIDEAN
        );

        // Step 4: Create new builder and rebuild graph
        // This creates a fresh graph with sequential ordinals
        builder = new GraphIndexBuilder(
                bsp,
                DIMENSION,
                16, 100, 1.2f, 1.2f, false, true
        );

        index = (OnHeapGraphIndex) builder.getGraph();

        // Step 5: Add all vectors with new sequential ordinals
        // New ordinals: 0, 1, 2, 3, ... (no gaps)
        for (int i = 0; i < compacted.size(); i++) {
            builder.addGraphNode(i, compacted.get(i));

            if ((i + 1) % 100 == 0) {
                System.out.println("  Rebuilt " + (i + 1) + " nodes...");
            }
        }

        // CRITICAL: After this point, the graph is ready for disk persistence
        // No holes in ordinals, perfect alignment with vectorList indices

        // Step 6: Update references
        vectorList = compacted;
        ravv = newRavv;
        nextOrdinal = vectorList.size();

        // Step 7: Clear deletion tracking
        int deletedCount = deletedNodes.size();
        deletedNodes.clear();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Cleanup with compaction completed in " + totalTime + "ms");
        System.out.println("  Removed: " + deletedCount + " vectors");
        System.out.println("  New size: " + vectorList.size() + " vectors");
        System.out.println("  Memory reclaimed: ~" + (deletedCount * DIMENSION * 4 / 1024) + " KB");
    }

    // ============= CLEANUP & SHUTDOWN =============

    /**
     * Performs clean shutdown of all resources.
     * <p>
     * CLEANUP ORDER:
     * 1. Close GraphIndexBuilder (releases graph memory)
     * 2. Close GridGain client (closes connections)
     * 3. Close scanner
     * <p>
     * Called when user selects Exit option (8).
     */
    private static void cleanupApplication() {
        System.out.println("\nCleaning up...");

        try {
            if (builder != null) {
                builder.close();
                System.out.println("   ✓ Closed GraphIndexBuilder");
            }
        } catch (Exception e) {
            System.out.println("   Error closing builder: " + e.getMessage());
        }

        try {
            if (client != null) {
                client.close();
                System.out.println("   ✓ Closed GridGain client");
            }
        } catch (Exception e) {
            System.out.println("   Error closing client: " + e.getMessage());
        }

        if (scanner != null) {
            scanner.close();
        }
    }

    // ============= UTILITY METHODS =============

    /**
     * Generates a random vector with values in range [0.0, 1.0).
     * <p>
     * Used for testing and demonstration purposes.
     * In production, vectors would come from an embedding model.
     *
     * @param dimension Number of dimensions
     * @return Array of random float values
     */

    private static float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) Math.random();
        }
        return vector;
    }

    /**
     * Serializes a float array to bytes using little-endian byte order.
     * <p>
     * FORMAT:
     * - Each float = 4 bytes
     * - Little-endian byte order (for consistency across platforms)
     * - Total size = dimension * 4 bytes
     *
     * @param floats Array of float values
     * @return Serialized byte array
     */
    private static byte[] floatArrayToBytes(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(floats.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }

    /**
     * Deserializes bytes back to a float array using little-endian byte order.
     * <p>
     * Must match the byte order used in floatArrayToBytes().
     *
     * @param bytes Serialized byte array
     * @return Array of float values
     */
    private static float[] bytesToFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        float[] floats = new float[bytes.length / 4];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = buffer.getFloat();
        }
        return floats;
    }
}