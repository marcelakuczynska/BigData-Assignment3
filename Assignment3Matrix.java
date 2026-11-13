import java.io.BufferedWriter;
import java.lang.management.ManagementFactory;
import java.lang.ref.Reference;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Assignment3Matrix {
    static final int[] SIZES = {128, 256, 512, 768, 1024};
    static final int[] THREADS = {1, 2, 4, 8};
    static final int WARMUPS = 2;
    static final int REPETITIONS = 5;
    static final long BASE_SEED = 20261113L;
    static final double MAX_ONE_RUN_MS = 30000;
    static final long MAX_ARRAY_BYTES = 256L * 1024 * 1024;
    static volatile long consumed;

    static class Setting {
        final String name;
        final int threads;
        final ThreadPoolExecutor pool;
        final List<Double> times = new ArrayList<>();
        final List<Double> memory = new ArrayList<>();

        Setting(String name, int threads, ThreadPoolExecutor pool) {
            this.name = name;
            this.threads = threads;
            this.pool = pool;
        }
    }

    static long[][] generate(int n, Random random) {
        long[][] a = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = random.nextInt(10);
            }
        }
        return a;
    }

    static void multiplyRows(long[][] a, long[][] b, long[][] c, int from, int to) {
        int n = a.length;
        for (int i = from; i < to; i++) {
            for (int k = 0; k < n; k++) {
                long x = a[i][k];
                for (int j = 0; j < n; j++) {
                    c[i][j] += x * b[k][j];
                }
            }
        }
    }

    static long[][] sequential(long[][] a, long[][] b) {
        long[][] c = new long[a.length][a.length];
        multiplyRows(a, b, c, 0, a.length);
        return c;
    }

    static long[][] parallel(long[][] a, long[][] b, int threads, ThreadPoolExecutor pool) throws Exception {
        int n = a.length;
        long[][] c = new long[n][n];
        List<Future<?>> tasks = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            int from = t * n / threads;
            int to = (t + 1) * n / threads;
            if (from < to) {
                tasks.add(pool.submit(() -> multiplyRows(a, b, c, from, to)));
            }
        }
        for (Future<?> task : tasks) {
            task.get();
        }
        return c;
    }

    static long[][] calculate(Setting setting, long[][] a, long[][] b) throws Exception {
        if (setting.threads == 0) return sequential(a, b);
        return parallel(a, b, setting.threads, setting.pool);
    }

    static void check(long[][] expected, long[][] actual) {
        if (expected.length != actual.length) throw new AssertionError("Different matrix size");
        for (int i = 0; i < expected.length; i++) {
            if (!Arrays.equals(expected[i], actual[i])) {
                throw new AssertionError("Different result in row " + i);
            }
        }
    }

    static void smallTests(Setting[] settings) throws Exception {
        long[][] a = {{1, 2}, {3, 4}};
        long[][] b = {{5, 6}, {7, 8}};
        long[][] expected = {{19, 22}, {43, 50}};
        long[][] identity = {{1, 0}, {0, 1}};
        long[][] zero = {{0, 0}, {0, 0}};
        for (Setting s : settings) {
            check(expected, calculate(s, a, b));
            check(a, calculate(s, a, identity));
            check(zero, calculate(s, a, zero));
        }
    }

    static long checksum(long[][] c) {
        int n = c.length;
        return c[0][0] + c[n / 2][n / 2] + c[n - 1][n - 1];
    }

    static double quantile(List<Double> numbers, double fraction) {
        List<Double> sorted = new ArrayList<>(numbers);
        Collections.sort(sorted);
        double index = (sorted.size() - 1) * fraction;
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        return sorted.get(lower) + (index - lower) * (sorted.get(upper) - sorted.get(lower));
    }

    static double median(List<Double> numbers) {
        return quantile(numbers, 0.5);
    }

    static double iqr(List<Double> numbers) {
        return quantile(numbers, 0.75) - quantile(numbers, 0.25);
    }

    static long usedHeap() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    static void gc() {
        System.gc();
        try {
            Thread.sleep(25);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    static Setting[] createSettings() {
        Setting[] settings = new Setting[THREADS.length + 1];
        settings[0] = new Setting("Sequential", 0, null);
        for (int i = 0; i < THREADS.length; i++) {
            int p = THREADS[i];
            ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(p);
            pool.prestartAllCoreThreads();
            settings[i + 1] = new Setting("Parallel", p, pool);
        }
        return settings;
    }

    static void shutdown(Setting[] settings) throws InterruptedException {
        for (Setting s : settings) {
            if (s.pool != null) s.pool.shutdown();
        }
        for (Setting s : settings) {
            if (s.pool != null) s.pool.awaitTermination(10, TimeUnit.SECONDS);
        }
    }

    static void runExperiment(int[] sizes, int warmups, int repeats, Path directory) throws Exception {
        Files.createDirectories(directory);
        Setting[] settings = createSettings();
        StringBuilder tests = new StringBuilder();
        StringBuilder report = new StringBuilder();
        try (BufferedWriter rawTime = Files.newBufferedWriter(directory.resolve("raw_times.csv"), StandardCharsets.UTF_8);
             BufferedWriter rawMemory = Files.newBufferedWriter(directory.resolve("raw_memory.csv"), StandardCharsets.UTF_8);
             BufferedWriter summary = Files.newBufferedWriter(directory.resolve("summary.csv"), StandardCharsets.UTF_8)) {
            rawTime.write("n,method,threads,repetition,time_ms,seed,numeric_type,loop_order,status\n");
            rawMemory.write("n,method,threads,repetition,heap_delta_mib,memory_metric,seed,numeric_type,loop_order\n");
            summary.write("n,method,threads,repetitions,median_ms,iqr_ms,median_heap_delta_mib,iqr_heap_delta_mib,estimated_three_matrices_mib,speedup,efficiency,seed\n");
            try {
                smallTests(settings);
                tests.append("Known 2x2, identity and zero matrices: 15/15 tests passed.\n");
                System.out.println("Correctness: small tests passed.");
                for (int n : sizes) {
                    long arrayBytes = 3L * n * n * Long.BYTES;
                    if (arrayBytes > MAX_ARRAY_BYTES) {
                        throw new IllegalArgumentException("Array budget exceeded for n=" + n);
                    }
                    long seed = BASE_SEED + n;
                    Random random = new Random(seed);
                    long[][] a = generate(n, random);
                    long[][] b = generate(n, random);
                    long[][] expected = sequential(a, b);
                    for (Setting s : settings) {
                        s.times.clear();
                        s.memory.clear();
                        check(expected, calculate(s, a, b));
                        for (int w = 0; w < warmups; w++) {
                            long[][] output = calculate(s, a, b);
                            check(expected, output);
                            consumed += checksum(output);
                        }
                    }
                    for (int r = 1; r <= repeats; r++) {
                        for (int offset = 0; offset < settings.length; offset++) {
                            Setting s = settings[(offset + r - 1) % settings.length];
                            long start = System.nanoTime();
                            long[][] output = calculate(s, a, b);
                            double ms = (System.nanoTime() - start) / 1_000_000.0;
                            check(expected, output);
                            consumed += checksum(output);
                            String status = ms > MAX_ONE_RUN_MS ? "over_budget" : "ok";
                            rawTime.write(String.format(Locale.US,
                                    "%d,%s,%d,%d,%.6f,%d,long64,i-k-j,%s%n",
                                    n, s.name, s.threads, r, ms, seed, status));
                            s.times.add(ms);
                            if (ms > MAX_ONE_RUN_MS) {
                                throw new IllegalStateException("A measured run exceeded 30 seconds for n=" + n);
                            }
                        }
                    }
                    for (Setting s : settings) {
                        for (int r = 1; r <= repeats; r++) {
                            gc();
                            long before = usedHeap();
                            long[][] output = calculate(s, a, b);
                            check(expected, output);
                            consumed += checksum(output);
                            gc();
                            long after = usedHeap();
                            double heapDelta = (after - before) / 1048576.0;
                            s.memory.add(heapDelta);
                            rawMemory.write(String.format(Locale.US,
                                    "%d,%s,%d,%d,%.6f,retained_java_heap_delta_after_gc,%d,long64,i-k-j%n",
                                    n, s.name, s.threads, r, heapDelta, seed));
                            Reference.reachabilityFence(output);
                        }
                    }
                    double tSeq = median(settings[0].times);
                    for (Setting s : settings) {
                        double medianMs = median(s.times);
                        double speedup = tSeq / medianMs;
                        double efficiency = s.threads == 0 ? 1.0 : speedup / s.threads;
                        summary.write(String.format(Locale.US,
                                "%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d%n",
                                n, s.name, s.threads, repeats, medianMs, iqr(s.times),
                                median(s.memory), iqr(s.memory), arrayBytes / 1048576.0,
                                speedup, efficiency, seed));
                        System.out.printf(Locale.US,
                                "n=%-4d %-10s threads=%d  median=%.4f ms  speedup=%.2f%n",
                                n, s.name, s.threads, medianMs, speedup);
                    }
                    tests.append("n=").append(n).append(": all parallel results equal sequential (including every repetition).\n");
                    rawTime.flush();
                    rawMemory.flush();
                    summary.flush();
                }
                report.append("COMPLETED\n");
            } finally {
                shutdown(settings);
            }
        } finally {
            Files.writeString(directory.resolve("correctness_tests.txt"), tests.toString(), StandardCharsets.UTF_8);
        }
        report.append("Java version: ").append(System.getProperty("java.version")).append("\n");
        report.append("OS: ").append(System.getProperty("os.name")).append("\n");
        report.append("Available logical processors in JVM: ").append(Runtime.getRuntime().availableProcessors()).append("\n");
        report.append("CPU: Intel Core i5-10210U (from Assignment 1 hardware record, same laptop)\n");
        report.append("Physical cores: 4; logical processors on laptop: 8; RAM: 7.72 GiB (Assignment 1 record)\n");
        report.append("JVM maximum heap MiB: ").append(Runtime.getRuntime().maxMemory() / 1048576).append("\n");
        report.append("Matrix sizes: ").append(Arrays.toString(sizes)).append("\n");
        report.append("Threads for parallel method: ").append(Arrays.toString(THREADS)).append("\n");
        report.append("Computational oversubscription: none planned (max 8 worker threads on 8 logical processors)\n");
        report.append("Processor affinity: not set; Windows default scheduler\n");
        report.append("Library worker threads: none (no BLAS or external matrix libraries)\n");
        report.append("JVM GC/JIT may use its own threads, not included in matrix worker count\n");
        report.append("Input generator: java.util.Random; seed=20261113+n; entries are integers 0..9; two matrices per n\n");
        report.append("Data: signed 64-bit long[][] row arrays; all methods use the same input matrices\n");
        report.append("Algorithm: i-k-j multiplication; same per-row computation in sequential and parallel versions\n");
        report.append("Parallel split: disjoint output row ranges, no simultaneous writes to the same row\n");
        report.append("Thread pools: created and prestarted before measurements; pool start-up excluded\n");
        report.append("Timer: System.nanoTime; output allocation and job submission/wait included; inputs and pool creation excluded\n");
        report.append("Warmups per configuration: ").append(warmups).append("; measured timing repetitions: ").append(repeats).append("\n");
        report.append("Memory: separate repeated approximate retained Java heap delta after System.gc(), not process RSS\n");
        report.append("Memory estimates: 3*n*n*8 bytes of primitive numeric array elements, excluding headers and thread stacks\n");
        report.append("Maximum allowed single multiplication duration: 30 seconds; fail if exceeded\n");
        report.append("Maximum budget for three numeric payload arrays: 256 MiB; fail if exceeded\n");
        report.append("Summary: median and interpolated 25th/75th percentile IQR; speedup Tseq/Tpar, efficiency speedup/p\n");
        report.append("The sequential kernel reuses the i-k-j loop-reordered implementation from Assignment 2\n");
        Files.writeString(directory.resolve("Experiment_Setup.txt"), report.toString(), StandardCharsets.UTF_8);
        System.out.println("DONE. Results saved in " + directory.toAbsolutePath());
        System.out.println("Checksum=" + consumed);
    }

    public static void main(String[] args) throws Exception {
        boolean quick = args.length > 0 && args[0].equals("--quick");
        runExperiment(quick ? new int[]{64, 128} : SIZES,
                quick ? 1 : WARMUPS,
                quick ? 2 : REPETITIONS,
                Path.of(quick ? "results_assignment3_quick" : "results_assignment3"));
    }
}
