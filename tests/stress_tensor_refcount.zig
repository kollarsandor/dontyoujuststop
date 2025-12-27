const std = @import("std");
const types = @import("types");
const Tensor = types.Tensor;
const Thread = std.Thread;
const Allocator = std.mem.Allocator;

const NUM_THREADS = 12;
const OPS_PER_THREAD = 15000;
const SHARED_TENSORS = 8;

const ThreadContext = struct {
    tensors: []Tensor,
    thread_id: usize,
    allocator: Allocator,
    barrier: *std.atomic.Atomic(usize),
    total_threads: usize,
};

fn threadWorker(ctx: ThreadContext) void {
    var prng = std.rand.DefaultPrng.init(@intCast(ctx.thread_id * 12345));
    const rand = prng.random();

    _ = ctx.barrier.fetchAdd(1, .SeqCst);
    while (ctx.barrier.load(.SeqCst) < ctx.total_threads) {
        std.Thread.yield() catch {};
    }

    var ops: usize = 0;
    while (ops < OPS_PER_THREAD) : (ops += 1) {
        const tensor_idx = rand.intRangeAtMost(usize, 0, SHARED_TENSORS - 1);
        const op_type = rand.intRangeAtMost(u8, 0, 99);

        if (op_type < 50) {
            ctx.tensors[tensor_idx].retain();

            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }

            ctx.tensors[tensor_idx].release();
        } else if (op_type < 75) {
            ctx.tensors[tensor_idx].retain();
            ctx.tensors[tensor_idx].retain();

            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }

            ctx.tensors[tensor_idx].release();
            ctx.tensors[tensor_idx].release();
        } else if (op_type < 90) {
            const other_idx = rand.intRangeAtMost(usize, 0, SHARED_TENSORS - 1);
            ctx.tensors[tensor_idx].retain();
            ctx.tensors[other_idx].retain();

            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }

            ctx.tensors[tensor_idx].release();
            ctx.tensors[other_idx].release();
        } else {
            var local_retains: usize = 0;
            while (local_retains < 5) : (local_retains += 1) {
                ctx.tensors[tensor_idx].retain();
            }

            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }

            local_retains = 0;
            while (local_retains < 5) : (local_retains += 1) {
                ctx.tensors[tensor_idx].release();
            }
        }

        if (ops % 1000 == 0 and ctx.thread_id == 0) {
            std.debug.print("Thread {d}: {d}/{d} operations completed\r", .{ ctx.thread_id, ops, OPS_PER_THREAD });
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .thread_safe = true,
    }){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("\n[LEAK DETECTED] Memory leaked!\n", .{});
            std.process.exit(1);
        }
    }
    const allocator = gpa.allocator();

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("TENSOR REFCOUNT STRESS TEST\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("Threads: {d}\n", .{NUM_THREADS});
    std.debug.print("Operations per thread: {d}\n", .{OPS_PER_THREAD});
    std.debug.print("Shared tensors: {d}\n", .{SHARED_TENSORS});
    std.debug.print("Total operations: {d}\n", .{NUM_THREADS * OPS_PER_THREAD});
    std.debug.print("-" ** 80 ++ "\n", .{});

    var tensors = try allocator.alloc(Tensor, SHARED_TENSORS);
    defer allocator.free(tensors);

    std.debug.print("Initializing {d} shared tensors...\n", .{SHARED_TENSORS});
    var i: usize = 0;
    while (i < SHARED_TENSORS) : (i += 1) {
        const shape = [_]usize{ 64, 64 };
        tensors[i] = try Tensor.init(allocator, &shape, f32);
    }

    std.debug.print("Verifying initial refcounts are 1...\n", .{});
    i = 0;
    while (i < SHARED_TENSORS) : (i += 1) {
        const refcount = @atomicLoad(usize, tensors[i].refcount, .SeqCst);
        if (refcount != 1) {
            std.debug.print("ERROR: Tensor {d} initial refcount is {d}, expected 1\n", .{ i, refcount });
            return error.InvalidInitialRefcount;
        }
    }
    std.debug.print("✓ All initial refcounts are correct\n\n", .{});

    var barrier = std.atomic.Atomic(usize).init(0);

    var threads = try allocator.alloc(Thread, NUM_THREADS);
    defer allocator.free(threads);

    std.debug.print("Spawning {d} threads...\n", .{NUM_THREADS});
    var timer = try std.time.Timer.start();
    const start = timer.read();

    var t: usize = 0;
    while (t < NUM_THREADS) : (t += 1) {
        const ctx = ThreadContext{
            .tensors = tensors,
            .thread_id = t,
            .allocator = allocator,
            .barrier = &barrier,
            .total_threads = NUM_THREADS,
        };
        threads[t] = try Thread.spawn(.{}, threadWorker, .{ctx});
    }

    std.debug.print("All threads spawned, executing concurrent operations...\n", .{});

    t = 0;
    while (t < NUM_THREADS) : (t += 1) {
        threads[t].join();
    }

    const end = timer.read();
    const elapsed_ns = end - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_ops = NUM_THREADS * OPS_PER_THREAD;
    const ops_per_sec = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);

    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("All threads completed successfully!\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});
    std.debug.print("Total time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("Total operations: {d}\n", .{total_ops});
    std.debug.print("Throughput: {d:.2} ops/sec\n", .{ops_per_sec});
    std.debug.print("Average time per operation: {d:.2} ns\n", .{@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_ops))});
    std.debug.print("-" ** 80 ++ "\n", .{});

    std.debug.print("Verifying final refcounts...\n", .{});
    var all_correct = true;
    i = 0;
    while (i < SHARED_TENSORS) : (i += 1) {
        const refcount = @atomicLoad(usize, tensors[i].refcount, .SeqCst);
        std.debug.print("  Tensor {d}: refcount = {d} ", .{ i, refcount });
        if (refcount == 1) {
            std.debug.print("✓\n", .{});
        } else {
            std.debug.print("✗ (expected 1)\n", .{});
            all_correct = false;
        }
    }

    if (!all_correct) {
        std.debug.print("\n[FAILED] Refcount validation failed!\n", .{});
        return error.RefcountMismatch;
    }

    std.debug.print("\n✓ All refcounts are correct (refcount == 1)\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});

    std.debug.print("Cleaning up tensors...\n", .{});
    i = 0;
    while (i < SHARED_TENSORS) : (i += 1) {
        tensors[i].deinit();
    }

    std.debug.print("\n[SUCCESS] Stress test passed! No memory leaks, no refcount errors.\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
}

test "concurrent tensor retain/release basic" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 8, 8 };
    var tensor = try Tensor.init(allocator, &shape, f32);
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 1), @atomicLoad(usize, tensor.refcount, .SeqCst));

    tensor.retain();
    try std.testing.expectEqual(@as(usize, 2), @atomicLoad(usize, tensor.refcount, .SeqCst));

    tensor.release();
    try std.testing.expectEqual(@as(usize, 1), @atomicLoad(usize, tensor.refcount, .SeqCst));
}

test "concurrent tensor multiple retains" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 4, 4 };
    var tensor = try Tensor.init(allocator, &shape, f32);
    defer tensor.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        tensor.retain();
    }
    try std.testing.expectEqual(@as(usize, 101), @atomicLoad(usize, tensor.refcount, .SeqCst));

    i = 0;
    while (i < 100) : (i += 1) {
        tensor.release();
    }
    try std.testing.expectEqual(@as(usize, 1), @atomicLoad(usize, tensor.refcount, .SeqCst));
}

test "concurrent tensor stress small scale" {
    const allocator = std.testing.allocator;

    const num_threads = 4;
    const num_tensors = 2;

    var tensors = try allocator.alloc(Tensor, num_tensors);
    defer allocator.free(tensors);

    var i: usize = 0;
    while (i < num_tensors) : (i += 1) {
        const shape = [_]usize{ 16, 16 };
        tensors[i] = try Tensor.init(allocator, &shape, f32);
    }

    var barrier = std.atomic.Atomic(usize).init(0);
    var threads = try allocator.alloc(Thread, num_threads);
    defer allocator.free(threads);

    var t: usize = 0;
    while (t < num_threads) : (t += 1) {
        const ctx = ThreadContext{
            .tensors = tensors,
            .thread_id = t,
            .allocator = allocator,
            .barrier = &barrier,
            .total_threads = num_threads,
        };
        threads[t] = try Thread.spawn(.{}, threadWorker, .{ctx});
    }

    t = 0;
    while (t < num_threads) : (t += 1) {
        threads[t].join();
    }

    i = 0;
    while (i < num_tensors) : (i += 1) {
        const refcount = @atomicLoad(usize, tensors[i].refcount, .SeqCst);
        try std.testing.expectEqual(@as(usize, 1), refcount);
        tensors[i].deinit();
    }
}
