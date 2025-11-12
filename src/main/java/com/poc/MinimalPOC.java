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
import java.util.ArrayList;
import java.util.List;

public class MinimalPOC {

    // THIS IS THE KEY - Get VectorTypeSupport for creating vectors
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void main(String[] args) throws Exception {
        System.out.println("=== JVector + GridGain 9 POC ===\n");

        // 1. Connect to existing GridGain cluster
        System.out.println("1. Connecting to GridGain cluster...");
        IgniteClient client = connectToCluster();

        // 2. Create table
        System.out.println("2. Creating table...");
        createTable(client);

        // 3. Insert test vectors
        System.out.println("3. Inserting test vectors...");
        int numVectors = 1000;
        insertTestVectors(client, numVectors);

        // 4. Build JVector index from GridGain
        System.out.println("4. Building JVector index from GridGain...");
        OnHeapGraphIndex index = buildJVectorIndexFromGridGain(client);

        // 5. Test search
        System.out.println("5. Testing vector search...");
        testVectorSearch(client, index);

        System.out.println("\n✓ POC Complete!");

        // Cleanup
        System.out.println("\nCleaning up...");
        client.close();
    }

    private static IgniteClient connectToCluster() {
        return IgniteClient.builder()
                .addresses("127.0.0.1:10800")
                .build();
    }

    private static void createTable(IgniteClient client) {
        try {
            client.sql().execute(null,
                    "CREATE ZONE IF NOT EXISTS poc_zone STORAGE PROFILES['default']");
        } catch (Exception e) {
            System.out.println("   Zone creation note: " + e.getMessage());
        }

        try {
            client.sql().execute(null,
                    "CREATE TABLE IF NOT EXISTS vectors (" +
                            "    id INT PRIMARY KEY," +
                            "    embedding VARBINARY" +
                            ") ZONE poc_zone");
        } catch (Exception e) {
            System.out.println("   Table creation note: " + e.getMessage());
        }
    }

    private static void insertTestVectors(IgniteClient client, int numVectors) {
        Table table = client.tables().table("vectors");

        for (int id = 0; id < numVectors; id++) {
            float[] vector = generateRandomVector(128);
            byte[] vectorBytes = floatArrayToBytes(vector);

            Tuple tuple = Tuple.create()
                    .set("id", id)
                    .set("embedding", vectorBytes);

            table.recordView().upsert(null, tuple);

            if ((id + 1) % 100 == 0) {
                System.out.println("   Inserted " + (id + 1) + " vectors...");
            }
        }
        System.out.println("   ✓ Inserted " + numVectors + " vectors to GridGain");
    }

    private static OnHeapGraphIndex buildJVectorIndexFromGridGain(IgniteClient client) throws Exception {
        int dimension = 128;

        // Step 1: Load ALL vectors from GridGain into a List
        System.out.println("   Loading vectors from GridGain...");
        List<VectorFloat<?>> vectorList = new ArrayList<>();

        try (ResultSet<SqlRow> rs = client.sql().execute(null,
                "SELECT id, embedding FROM vectors ORDER BY id")) {
            while (rs.hasNext()) {
                SqlRow row = rs.next();
                byte[] vectorBytes = row.value("embedding");
                float[] floats = bytesToFloatArray(vectorBytes);

                // CORRECT WAY TO CREATE VECTORFLOAT
                VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
                for (int i = 0; i < dimension; i++) {
                    vector.set(i, floats[i]);
                }
                vectorList.add(vector);
            }
        }
        System.out.println("   Loaded " + vectorList.size() + " vectors");

        // Step 2: Wrap in RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectorList, dimension);

        // Step 3: Create BuildScoreProvider
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
                ravv,
                VectorSimilarityFunction.EUCLIDEAN
        );

        // Step 4: Create GraphIndexBuilder and build the index
        System.out.println("   Building HNSW graph index...");
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
                bsp,
                dimension,      // dimension
                16,             // M (max connections per node)
                100,            // efConstruction (beam width)
                1.2f,           // neighborOverflow
                1.2f,           // alpha (diversity)
                false,          // addHierarchy
                true            // refineFinalGraph
        )) {
            // THIS IS THE KEY METHOD - build() creates the complete index!
            OnHeapGraphIndex index = (OnHeapGraphIndex) builder.build(ravv);
            System.out.println("   ✓ Index built successfully");
            return index;
        }
    }

    private static void testVectorSearch(IgniteClient client, OnHeapGraphIndex index) throws Exception {
        int dimension = 128;

        // Get a vector from GridGain to use as query (ID 42)
        byte[] queryBytes;
        try (ResultSet<SqlRow> rs = client.sql().execute(null,
                "SELECT embedding FROM vectors WHERE id = 42")) {
            SqlRow row = rs.next();
            queryBytes = row.value("embedding");
        }

        float[] queryFloats = bytesToFloatArray(queryBytes);
        VectorFloat<?> queryVector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            queryVector.set(i, queryFloats[i]);
        }

        // Load all vectors for scoring (needed for DefaultSearchScoreProvider)
        List<VectorFloat<?>> allVectors = new ArrayList<>();
        try (ResultSet<SqlRow> rs = client.sql().execute(null,
                "SELECT embedding FROM vectors ORDER BY id")) {
            while (rs.hasNext()) {
                SqlRow row = rs.next();
                byte[] vectorBytes = row.value("embedding");
                float[] floats = bytesToFloatArray(vectorBytes);

                VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
                for (int i = 0; i < dimension; i++) {
                    vector.set(i, floats[i]);
                }
                allVectors.add(vector);
            }
        }
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(allVectors, dimension);

        // Create SearchScoreProvider
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(
                queryVector,
                VectorSimilarityFunction.EUCLIDEAN,
                ravv
        );

        // Perform search
        int topK = 10;
        try (GraphSearcher searcher = new GraphSearcher(index)) {
            SearchResult result = searcher.search(ssp, topK, Bits.ALL);

            System.out.println("   ✓ Search results for vector ID 42:");
            System.out.println("      Top " + topK + " nearest neighbors:");

            for (SearchResult.NodeScore ns : result.getNodes()) {
                System.out.printf("        ID: %d, Score: %.4f%n", ns.node, ns.score);
            }
        }
    }

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
}