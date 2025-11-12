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

public class MinimalPOC_Incremental {

    // ============= CLASS-LEVEL STATE (Keep Alive) =============
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int DIMENSION = 128;

    // Core components that stay alive
    private static IgniteClient client;
    private static GraphIndexBuilder builder;
    private static OnHeapGraphIndex index;
    private static List<VectorFloat<?>> vectorList;
    private static RandomAccessVectorValues ravv;

    // Track next available ordinal
    private static int nextOrdinal = 0;

    // Scanner for user input
    private static Scanner scanner;

    // Add deletion tracking
    private static Set<Integer> deletedNodes = new HashSet<>();
    private static boolean needsGraphCleanup = false;  // NEW


    // ============= MAIN =============
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

    // ============= INITIALIZATION (Runs Once) =============

    private static void initializeConnection() {
        System.out.println("1. Connecting to GridGain cluster...");
        client = IgniteClient.builder()
                .addresses("127.0.0.1:10800")
                .build();
        scanner = new Scanner(System.in);
        System.out.println("   ✓ Connected\n");
    }

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

                VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(DIMENSION);
                for (int i = 0; i < DIMENSION; i++) {
                    vector.set(i, floats[i]);
                }
                vectorList.add(vector);
            }
        }
        System.out.println("   Loaded " + vectorList.size() + " vectors");
        nextOrdinal = vectorList.size();

        // Create RAVV
        ravv = new ListRandomAccessVectorValues(vectorList, DIMENSION);

        // Create BuildScoreProvider
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
                ravv,
                VectorSimilarityFunction.EUCLIDEAN
        );

        // Create builder
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
        index = (OnHeapGraphIndex) builder.getGraph();

        // Now add all nodes incrementally, just like in their example
        for (int i = 0; i < vectorList.size(); i++) {
            builder.addGraphNode(i, vectorList.get(i));

            if ((i + 1) % 100 == 0) {
                System.out.println("   Added " + (i + 1) + " nodes to graph");
            }
        }

        // Call cleanup like they do
//        builder.cleanup();

        System.out.println("   ✓ Index built with " + index.size() + " nodes");

        // Analyze connectivity
        System.out.println("\n   Analyzing graph connectivity:");
        analyzeGraphConnectivity();
    }

    // ============= INTERACTIVE LOOP =============

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

    private static int getIntInput() {
        try {
            return scanner.nextInt();
        } catch (Exception e) {
            scanner.nextLine(); // Clear buffer
            return -1;
        }
    }

    // ============= INCREMENTAL ADD (The Key Method!) =============

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

    // ============= SEARCH (On Demand) =============

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

    // ============= STATISTICS =============

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

    // ============= CLEANUP =============

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

    // ============= DELETION METHODS =============

    private static void markVectorDeleted() {
        System.out.print("Enter vector ID to delete: ");
        int deleteId = getIntInput();

        if (deleteId < 0 || deleteId >= nextOrdinal) {
            System.out.println("Invalid ID.\n");
            return;
        }

        builder.markNodeDeleted(deleteId);
        deletedNodes.add(deleteId);
        needsGraphCleanup = true;  // NEW

        System.out.println("✓ Marked for deletion");
        System.out.println("  Total marked: " + deletedNodes.size());
        System.out.println("💡 Cleanup when convenient via menu option 6 or 7");
    }
    private static void performCleanup() throws Exception {
        if (!needsGraphCleanup) {
            System.out.println("No marked deletions to cleanup");
            return;
        }

        System.out.println("⚠️ Cleanup blocks operations - ensure no concurrent activity!");
        System.out.println("Cleaning " + deletedNodes.size() + " deleted nodes from graph...");

        long startTime = System.currentTimeMillis();
        builder.cleanup();  // Removes from graph structure
        needsGraphCleanup = false;  // Graph is now clean

        System.out.println("✓ Cleanup done in " + (System.currentTimeMillis() - startTime) + "ms");
        System.out.println("  Graph structure optimized");
        System.out.println("  ⚠️ " + deletedNodes.size() + " vectors still occupy ~" +
                (deletedNodes.size() * DIMENSION * 4 / 1024) + " KB in vectorList");
        System.out.println("  💡 Use option 7 to compact vectorList and reclaim memory");

        // DON'T clear deletedNodes! We need it for compaction!
    }

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
        builder = new GraphIndexBuilder(
                bsp,
                DIMENSION,
                16, 100, 1.2f, 1.2f, false, true
        );

        index = (OnHeapGraphIndex) builder.getGraph();

        // Step 5: Add all vectors with new sequential ordinals
        for (int i = 0; i < compacted.size(); i++) {
            builder.addGraphNode(i, compacted.get(i));

            if ((i + 1) % 100 == 0) {
                System.out.println("  Rebuilt " + (i + 1) + " nodes...");
            }
        }

        // CAN NOW BE SAVED

        // Step 6: Update references
        vectorList = compacted;
        ravv = newRavv;
        nextOrdinal = vectorList.size();

        // NOW clear deletedNodes - memory fully reclaimed
        int deletedCount = deletedNodes.size();
        deletedNodes.clear();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Cleanup with compaction completed in " + totalTime + "ms");
        System.out.println("  Removed: " + deletedCount + " vectors");
        System.out.println("  New size: " + vectorList.size() + " vectors");
        System.out.println("  Memory reclaimed: ~" + (deletedCount * DIMENSION * 4 / 1024) + " KB");
    }

    // ============= UTILITY METHODS =============

    private static float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) Math.random();
        }
        return vector;
    }

    private static byte[] floatArrayToBytes(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(floats.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }

    private static float[] bytesToFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        float[] floats = new float[bytes.length / 4];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = buffer.getFloat();
        }
        return floats;
    }

    // After adding each incremental vector, check:
// - How many connections does the new node have?
// - Did existing nodes get new connections to this node?
// - Track average connectivity over time
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



}