const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const quantum_logic = @import("quantum_logic.zig");

const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;
const ArrayList = std.ArrayList;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;

var global_gpa = std.heap.GeneralPurposeAllocator(.{}){};
const global_allocator: Allocator = global_gpa.allocator();

pub const JAIDE_SUCCESS: c_int = 0;
pub const JAIDE_ERROR_NULL_POINTER: c_int = -1;
pub const JAIDE_ERROR_ALLOCATION: c_int = -2;
pub const JAIDE_ERROR_NODE_NOT_FOUND: c_int = -3;
pub const JAIDE_ERROR_EDGE_NOT_FOUND: c_int = -4;
pub const JAIDE_ERROR_INVALID_QUALITY: c_int = -5;
pub const JAIDE_ERROR_OPTIMIZATION_FAILED: c_int = -6;
pub const JAIDE_ERROR_INVALID_STRING: c_int = -7;
pub const JAIDE_ERROR_OPERATION_FAILED: c_int = -8;

pub const CQuantumState = extern struct {
    real: f64,
    imag: f64,
};

pub const CNode = extern struct {
    id: [*c]const u8,
    id_len: usize,
    type_name: [*c]const u8,
    type_name_len: usize,
    quantum_real: f64,
    quantum_imag: f64,
    phase: f64,
};

pub const CEdge = extern struct {
    source: [*c]const u8,
    source_len: usize,
    target: [*c]const u8,
    target_len: usize,
    weight: f64,
    quality: c_int,
    quantum_real: f64,
    quantum_imag: f64,
    fractal_dimension: f64,
};

pub const CGraph = opaque {
    pub fn fromInternal(graph: *SelfSimilarRelationalGraph) *CGraph {
        return @ptrCast(graph);
    }

    pub fn toInternal(self: *CGraph) *SelfSimilarRelationalGraph {
        return @ptrCast(@alignCast(self));
    }

    pub fn toInternalConst(self: *const CGraph) *const SelfSimilarRelationalGraph {
        return @ptrCast(@alignCast(self));
    }
};

pub const OptimizationStatistics = struct {
    iterations_completed: usize,
    moves_accepted: usize,
    moves_rejected: usize,
    best_energy: f64,
    current_energy: f64,
    temperature: f64,
    acceptance_rate: f64,

    pub fn init() OptimizationStatistics {
        return OptimizationStatistics{
            .iterations_completed = 0,
            .moves_accepted = 0,
            .moves_rejected = 0,
            .best_energy = std.math.inf(f64),
            .current_energy = std.math.inf(f64),
            .temperature = 0.0,
            .acceptance_rate = 0.0,
        };
    }

    pub fn updateAcceptanceRate(self: *OptimizationStatistics) void {
        const total_moves = self.moves_accepted + self.moves_rejected;
        if (total_moves > 0) {
            self.acceptance_rate = @as(f64, @floatFromInt(self.moves_accepted)) / @as(f64, @floatFromInt(total_moves));
        }
    }
};

pub const EntangledStochasticSymmetryOptimizer = struct {
    temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    current_iteration: usize,
    min_temperature: f64,
    initial_temperature: f64,
    reheat_threshold: f64,
    reheat_factor: f64,
    adaptive_cooling: bool,
    prng: std.rand.DefaultPrng,
    allocator: Allocator,
    statistics: OptimizationStatistics,

    const Self = @This();
    const DEFAULT_MIN_TEMP: f64 = 0.001;
    const DEFAULT_REHEAT_THRESHOLD: f64 = 0.1;
    const DEFAULT_REHEAT_FACTOR: f64 = 2.0;

    pub fn init(
        allocator: Allocator,
        initial_temp: f64,
        cooling_rate: f64,
        max_iterations: usize,
    ) Self {
        const seed = @as(u64, @bitCast(std.time.milliTimestamp()));
        return Self{
            .temperature = initial_temp,
            .cooling_rate = cooling_rate,
            .max_iterations = max_iterations,
            .current_iteration = 0,
            .min_temperature = DEFAULT_MIN_TEMP,
            .initial_temperature = initial_temp,
            .reheat_threshold = DEFAULT_REHEAT_THRESHOLD,
            .reheat_factor = DEFAULT_REHEAT_FACTOR,
            .adaptive_cooling = true,
            .prng = std.rand.DefaultPrng.init(seed),
            .allocator = allocator,
            .statistics = OptimizationStatistics.init(),
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn setMinTemperature(self: *Self, min_temp: f64) void {
        self.min_temperature = min_temp;
    }

    pub fn setAdaptiveCooling(self: *Self, enabled: bool) void {
        self.adaptive_cooling = enabled;
    }

    pub fn setReheatParameters(self: *Self, threshold: f64, factor: f64) void {
        self.reheat_threshold = threshold;
        self.reheat_factor = factor;
    }

    fn computeEnergy(graph: *const SelfSimilarRelationalGraph) f64 {
        var energy: f64 = 0.0;

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                energy += edge.weight * edge.fractal_dimension;
                energy += edge.quantum_correlation.magnitude();
            }
        }

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            energy += entry.value_ptr.quantum_state.magnitude();
        }

        return energy;
    }

    fn cloneGraph(self: *Self, source: *const SelfSimilarRelationalGraph) !*SelfSimilarRelationalGraph {
        const new_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        new_graph.* = SelfSimilarRelationalGraph.init(self.allocator);
        errdefer {
            new_graph.deinit();
            self.allocator.destroy(new_graph);
        }

        var node_iter = source.nodes.iterator();
        while (node_iter.next()) |entry| {
            var cloned_node = try entry.value_ptr.clone(self.allocator);
            try new_graph.addNode(cloned_node);
        }

        var edge_iter = source.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                var cloned_edge = try edge.clone(self.allocator);
                try new_graph.addEdge(cloned_edge);
            }
        }

        new_graph.fractal_depth = source.fractal_depth;
        @memcpy(&new_graph.topology_hash, &source.topology_hash);

        return new_graph;
    }

    fn perturbGraph(self: *Self, graph: *SelfSimilarRelationalGraph) void {
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                if (self.prng.random().float(f64) < 0.3) {
                    const perturbation = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.1;
                    edge.weight = @max(0.0, @min(1.0, edge.weight + perturbation));
                }
            }
        }

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            if (self.prng.random().float(f64) < 0.2) {
                const state = entry.value_ptr.quantum_state;
                const angle = self.prng.random().float(f64) * 2.0 * std.math.pi;
                const perturbation = self.temperature * 0.05;
                const new_real = state.re + perturbation * @cos(angle);
                const new_imag = state.im + perturbation * @sin(angle);
                entry.value_ptr.quantum_state = Complex(f64).init(new_real, new_imag);
            }
        }
    }

    fn acceptMove(self: *Self, delta_energy: f64) bool {
        if (delta_energy <= 0.0) {
            return true;
        }
        if (self.temperature < 1e-10) {
            return false;
        }
        const acceptance_probability = @exp(-delta_energy / self.temperature);
        const random_value = self.prng.random().float(f64);
        return random_value < acceptance_probability;
    }

    fn coolTemperature(self: *Self) void {
        if (self.adaptive_cooling) {
            var adjusted_rate = self.cooling_rate;
            if (self.statistics.acceptance_rate > 0.6) {
                adjusted_rate *= 0.98;
            } else if (self.statistics.acceptance_rate < 0.2) {
                adjusted_rate *= 1.02;
            }
            adjusted_rate = @max(0.8, @min(0.999, adjusted_rate));
            self.temperature *= adjusted_rate;
        } else {
            self.temperature *= self.cooling_rate;
        }
        if (self.temperature < self.min_temperature) {
            self.temperature = self.min_temperature;
        }
    }

    pub fn optimize(self: *Self, graph: *SelfSimilarRelationalGraph) !void {
        self.statistics = OptimizationStatistics.init();
        self.temperature = self.initial_temperature;
        self.current_iteration = 0;

        var current_energy = computeEnergy(graph);
        var best_energy = current_energy;
        self.statistics.best_energy = best_energy;
        self.statistics.current_energy = current_energy;

        var stagnation_counter: usize = 0;

        while (self.current_iteration < self.max_iterations) : (self.current_iteration += 1) {
            const candidate = try self.cloneGraph(graph);
            defer {
                candidate.deinit();
                self.allocator.destroy(candidate);
            }

            self.perturbGraph(candidate);
            const candidate_energy = computeEnergy(candidate);
            const delta_energy = candidate_energy - current_energy;

            if (self.acceptMove(delta_energy)) {
                graph.clear();
                var node_iter = candidate.nodes.iterator();
                while (node_iter.next()) |entry| {
                    var cloned_node = try entry.value_ptr.clone(self.allocator);
                    try graph.addNode(cloned_node);
                }
                var edge_iter = candidate.edges.iterator();
                while (edge_iter.next()) |entry| {
                    for (entry.value_ptr.items) |*edge| {
                        var cloned_edge = try edge.clone(self.allocator);
                        try graph.addEdge(cloned_edge);
                    }
                }

                current_energy = candidate_energy;
                self.statistics.moves_accepted += 1;

                if (candidate_energy < best_energy) {
                    best_energy = candidate_energy;
                    self.statistics.best_energy = best_energy;
                    stagnation_counter = 0;
                }
            } else {
                self.statistics.moves_rejected += 1;
                stagnation_counter += 1;
            }

            self.statistics.current_energy = current_energy;
            self.statistics.temperature = self.temperature;
            self.statistics.iterations_completed = self.current_iteration + 1;
            self.statistics.updateAcceptanceRate();

            self.coolTemperature();

            if (stagnation_counter > self.max_iterations / 10) {
                if (self.temperature < self.initial_temperature * self.reheat_threshold) {
                    self.temperature = self.temperature * self.reheat_factor;
                    stagnation_counter = 0;
                }
            }

            if (self.temperature < self.min_temperature) {
                break;
            }
        }
    }
};

pub const COptimizer = opaque {
    pub fn fromInternal(opt: *EntangledStochasticSymmetryOptimizer) *COptimizer {
        return @ptrCast(opt);
    }

    pub fn toInternal(self: *COptimizer) *EntangledStochasticSymmetryOptimizer {
        return @ptrCast(@alignCast(self));
    }
};

fn cStringToSlice(ptr: [*c]const u8) ?[]const u8 {
    if (ptr == null) return null;
    var len: usize = 0;
    while (ptr[len] != 0) : (len += 1) {}
    if (len == 0) return null;
    return ptr[0..len];
}

pub export fn jaide_create_graph() callconv(.C) ?*CGraph {
    const graph = global_allocator.create(SelfSimilarRelationalGraph) catch return null;
    graph.* = SelfSimilarRelationalGraph.init(global_allocator);
    return CGraph.fromInternal(graph);
}

pub export fn jaide_destroy_graph(graph: ?*CGraph) callconv(.C) void {
    if (graph) |g| {
        const internal = g.toInternal();
        internal.deinit();
        global_allocator.destroy(internal);
    }
}

pub export fn jaide_graph_node_count(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return 0;
    const internal = graph.?.toInternal();
    return @intCast(internal.nodeCount());
}

pub export fn jaide_graph_edge_count(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return 0;
    const internal = graph.?.toInternal();
    return @intCast(internal.edgeCount());
}

pub export fn jaide_add_node(
    graph: ?*CGraph,
    id: [*c]const u8,
    type_name: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const type_slice = cStringToSlice(type_name) orelse "";

    const internal = graph.?.toInternal();

    var node = Node.init(
        global_allocator,
        id_slice,
        type_slice,
        1.0,
        0.0,
        0.0,
    ) catch return JAIDE_ERROR_ALLOCATION;

    internal.addNode(node) catch {
        node.deinit();
        return JAIDE_ERROR_OPERATION_FAILED;
    };

    return JAIDE_SUCCESS;
}

pub export fn jaide_remove_node(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) {
        return JAIDE_ERROR_NODE_NOT_FOUND;
    }

    if (internal.nodes.fetchRemove(id_slice)) |removed| {
        global_allocator.free(removed.key);
        var node = removed.value;
        node.deinit();
    }

    if (internal.quantum_register.fetchRemove(id_slice)) |removed| {
        global_allocator.free(removed.key);
    }

    var edges_to_remove = ArrayList(EdgeKey).init(global_allocator);
    defer edges_to_remove.deinit();

    var edge_iter = internal.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key.source, id_slice) or std.mem.eql(u8, key.target, id_slice)) {
            edges_to_remove.append(key) catch continue;
        }
    }

    for (edges_to_remove.items) |key| {
        if (internal.edges.fetchRemove(key)) |removed| {
            for (removed.value.items) |*edge| {
                edge.deinit();
            }
            var list = removed.value;
            list.deinit();
        }
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_node_quantum_state(
    graph: ?*CGraph,
    id: [*c]const u8,
    out_state: ?*CQuantumState,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (out_state == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;

    out_state.?.real = node.quantum_state.re;
    out_state.?.imag = node.quantum_state.im;

    return JAIDE_SUCCESS;
}

pub export fn jaide_set_node_quantum_state(
    graph: ?*CGraph,
    id: [*c]const u8,
    real: f64,
    imag: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    node.quantum_state = Complex(f64).init(real, imag);

    internal.setQuantumState(id_slice, Complex(f64).init(real, imag)) catch {
        return JAIDE_ERROR_OPERATION_FAILED;
    };

    return JAIDE_SUCCESS;
}

pub export fn jaide_add_edge(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    weight: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const source_slice = cStringToSlice(source) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToSlice(target) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    if (!internal.hasNode(source_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;
    if (!internal.hasNode(target_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    var edge = Edge.init(
        global_allocator,
        source_slice,
        target_slice,
        .coherent,
        weight,
        1.0,
        0.0,
        1.0,
    ) catch return JAIDE_ERROR_ALLOCATION;

    internal.addEdge(edge) catch {
        edge.deinit();
        return JAIDE_ERROR_OPERATION_FAILED;
    };

    return JAIDE_SUCCESS;
}

pub export fn jaide_remove_edge(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const source_slice = cStringToSlice(source) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToSlice(target) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    if (!internal.hasEdge(source_slice, target_slice)) {
        return JAIDE_ERROR_EDGE_NOT_FOUND;
    }

    const key = EdgeKey{ .source = source_slice, .target = target_slice };
    if (internal.edges.fetchRemove(key)) |removed| {
        for (removed.value.items) |*edge| {
            edge.deinit();
        }
        var list = removed.value;
        list.deinit();
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_edge_weight(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const source_slice = cStringToSlice(source) orelse return -1.0;
    const target_slice = cStringToSlice(target) orelse return -1.0;

    const internal = graph.?.toInternal();

    const edges = internal.getEdgeList(source_slice, target_slice);
    if (edges.len == 0) return -1.0;

    return edges[0].weight;
}

pub export fn jaide_set_edge_quality(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    quality: c_int,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const source_slice = cStringToSlice(source) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToSlice(target) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    const edges = internal.getEdges(source_slice, target_slice) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;

    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;

    const new_quality: EdgeQuality = switch (quality) {
        0 => .superposition,
        1 => .entangled,
        2 => .coherent,
        3 => .collapsed,
        4 => .fractal,
        else => return JAIDE_ERROR_INVALID_QUALITY,
    };

    for (edges.items) |*edge| {
        edge.quality = new_quality;
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_apply_hadamard(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.applyQuantumGate(id_slice, nsir_core.hadamardGate);

    return JAIDE_SUCCESS;
}

pub export fn jaide_apply_pauli_x(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.applyQuantumGate(id_slice, nsir_core.pauliXGate);

    return JAIDE_SUCCESS;
}

pub export fn jaide_apply_pauli_y(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.applyQuantumGate(id_slice, nsir_core.pauliYGate);

    return JAIDE_SUCCESS;
}

pub export fn jaide_apply_pauli_z(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.applyQuantumGate(id_slice, nsir_core.pauliZGate);

    return JAIDE_SUCCESS;
}

pub export fn jaide_entangle_nodes(
    graph: ?*CGraph,
    node1: [*c]const u8,
    node2: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const node1_slice = cStringToSlice(node1) orelse return JAIDE_ERROR_INVALID_STRING;
    const node2_slice = cStringToSlice(node2) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    if (!internal.hasNode(node1_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;
    if (!internal.hasNode(node2_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.entangleNodes(node1_slice, node2_slice) catch {
        return JAIDE_ERROR_OPERATION_FAILED;
    };

    return JAIDE_SUCCESS;
}

pub export fn jaide_measure_node(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const id_slice = cStringToSlice(node_id) orelse return -1.0;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return -1.0;

    return internal.measure(id_slice);
}

pub export fn jaide_create_optimizer(
    temp: f64,
    cooling: f64,
    max_iter: c_int,
) callconv(.C) ?*COptimizer {
    if (max_iter <= 0) return null;

    const opt = global_allocator.create(EntangledStochasticSymmetryOptimizer) catch return null;
    opt.* = EntangledStochasticSymmetryOptimizer.init(
        global_allocator,
        temp,
        cooling,
        @intCast(max_iter),
    );
    return COptimizer.fromInternal(opt);
}

pub export fn jaide_destroy_optimizer(opt: ?*COptimizer) callconv(.C) void {
    if (opt) |o| {
        const internal = o.toInternal();
        internal.deinit();
        global_allocator.destroy(internal);
    }
}

pub export fn jaide_optimize_graph(
    opt: ?*COptimizer,
    graph: ?*CGraph,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const internal_opt = opt.?.toInternal();
    const internal_graph = graph.?.toInternal();

    internal_opt.optimize(internal_graph) catch {
        return JAIDE_ERROR_OPTIMIZATION_FAILED;
    };

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_fractal_dimension(graph: ?*CGraph) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const internal = graph.?.toInternal();
    return internal.calculateFractalDimension();
}

pub export fn jaide_get_topology_hash(
    graph: ?*CGraph,
    out_hash: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (out_hash == null) return JAIDE_ERROR_NULL_POINTER;

    const internal = graph.?.toInternalConst();
    const hash_hex = internal.getTopologyHashHex();

    const copy_len = @min(hash_hex.len, max_len - 1);
    @memcpy(out_hash[0..copy_len], hash_hex[0..copy_len]);
    out_hash[copy_len] = 0;

    return JAIDE_SUCCESS;
}

pub export fn jaide_encode_information(
    graph: ?*CGraph,
    data: [*c]const u8,
    out_node_id: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (data == null) return JAIDE_ERROR_NULL_POINTER;
    if (out_node_id == null) return JAIDE_ERROR_NULL_POINTER;

    const data_slice = cStringToSlice(data) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const node_id = internal.encodeInformation(data_slice) catch {
        return JAIDE_ERROR_OPERATION_FAILED;
    };
    defer global_allocator.free(node_id);

    const copy_len = @min(node_id.len, max_len - 1);
    @memcpy(out_node_id[0..copy_len], node_id[0..copy_len]);
    out_node_id[copy_len] = 0;

    return JAIDE_SUCCESS;
}

pub export fn jaide_decode_information(
    graph: ?*CGraph,
    node_id: [*c]const u8,
    out_data: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (node_id == null) return JAIDE_ERROR_NULL_POINTER;
    if (out_data == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const data = internal.decodeInformation(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;

    const copy_len = @min(data.len, max_len - 1);
    @memcpy(out_data[0..copy_len], data[0..copy_len]);
    out_data[copy_len] = 0;

    return JAIDE_SUCCESS;
}

pub export fn jaide_has_node(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return 0;

    const id_slice = cStringToSlice(id) orelse return 0;
    const internal = graph.?.toInternalConst();

    return if (internal.hasNode(id_slice)) 1 else 0;
}

pub export fn jaide_has_edge(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return 0;

    const source_slice = cStringToSlice(source) orelse return 0;
    const target_slice = cStringToSlice(target) orelse return 0;
    const internal = graph.?.toInternalConst();

    return if (internal.hasEdge(source_slice, target_slice)) 1 else 0;
}

pub export fn jaide_clear_graph(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = graph.?.toInternal();
    internal.clear();
    return JAIDE_SUCCESS;
}

pub export fn jaide_get_node_phase(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const id_slice = cStringToSlice(id) orelse return -1.0;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return -1.0;
    return node.phase;
}

pub export fn jaide_set_node_phase(
    graph: ?*CGraph,
    id: [*c]const u8,
    phase: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    node.phase = phase;

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_edge_quality(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return -1;

    const source_slice = cStringToSlice(source) orelse return -1;
    const target_slice = cStringToSlice(target) orelse return -1;

    const internal = graph.?.toInternal();
    const edges = internal.getEdgeList(source_slice, target_slice);
    if (edges.len == 0) return -1;

    return @intFromEnum(edges[0].quality);
}

pub export fn jaide_get_node_data(
    graph: ?*CGraph,
    id: [*c]const u8,
    out_data: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (id == null) return JAIDE_ERROR_NULL_POINTER;
    if (out_data == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;

    const copy_len = @min(node.data.len, max_len - 1);
    @memcpy(out_data[0..copy_len], node.data[0..copy_len]);
    out_data[copy_len] = 0;

    return JAIDE_SUCCESS;
}

pub export fn jaide_set_edge_weight(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    weight: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const source_slice = cStringToSlice(source) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToSlice(target) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    const edges = internal.getEdges(source_slice, target_slice) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;
    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;

    for (edges.items) |*edge| {
        edge.weight = weight;
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_edge_fractal_dimension(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const source_slice = cStringToSlice(source) orelse return -1.0;
    const target_slice = cStringToSlice(target) orelse return -1.0;

    const internal = graph.?.toInternal();
    const edges = internal.getEdgeList(source_slice, target_slice);
    if (edges.len == 0) return -1.0;

    return edges[0].fractal_dimension;
}

pub export fn jaide_set_edge_fractal_dimension(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    dimension: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const source_slice = cStringToSlice(source) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToSlice(target) orelse return JAIDE_ERROR_INVALID_STRING;

    const internal = graph.?.toInternal();

    const edges = internal.getEdges(source_slice, target_slice) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;
    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;

    for (edges.items) |*edge| {
        edge.fractal_dimension = dimension;
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_set_optimizer_min_temperature(
    opt: ?*COptimizer,
    min_temp: f64,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();
    internal.setMinTemperature(min_temp);
    return JAIDE_SUCCESS;
}

pub export fn jaide_set_optimizer_adaptive_cooling(
    opt: ?*COptimizer,
    enabled: c_int,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();
    internal.setAdaptiveCooling(enabled != 0);
    return JAIDE_SUCCESS;
}

pub export fn jaide_set_optimizer_reheat_params(
    opt: ?*COptimizer,
    threshold: f64,
    factor: f64,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();
    internal.setReheatParameters(threshold, factor);
    return JAIDE_SUCCESS;
}

pub export fn jaide_get_optimizer_statistics(
    opt: ?*COptimizer,
    out_iterations: ?*c_int,
    out_best_energy: ?*f64,
    out_acceptance_rate: ?*f64,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();

    if (out_iterations) |ptr| {
        ptr.* = @intCast(internal.statistics.iterations_completed);
    }
    if (out_best_energy) |ptr| {
        ptr.* = internal.statistics.best_energy;
    }
    if (out_acceptance_rate) |ptr| {
        ptr.* = internal.statistics.acceptance_rate;
    }

    return JAIDE_SUCCESS;
}

pub export fn jaide_get_node_probability(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const id_slice = cStringToSlice(id) orelse return -1.0;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return -1.0;
    return node.probability();
}

pub export fn jaide_get_node_magnitude(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const id_slice = cStringToSlice(id) orelse return -1.0;
    const internal = graph.?.toInternal();

    const node = internal.getNode(id_slice) orelse return -1.0;
    return node.magnitude();
}

pub export fn jaide_get_edge_correlation_magnitude(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;

    const source_slice = cStringToSlice(source) orelse return -1.0;
    const target_slice = cStringToSlice(target) orelse return -1.0;

    const internal = graph.?.toInternal();
    const edges = internal.getEdgeList(source_slice, target_slice);
    if (edges.len == 0) return -1.0;

    return edges[0].correlationMagnitude();
}

pub export fn jaide_apply_identity_gate(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;

    const id_slice = cStringToSlice(node_id) orelse return JAIDE_ERROR_INVALID_STRING;
    const internal = graph.?.toInternal();

    if (!internal.hasNode(id_slice)) return JAIDE_ERROR_NODE_NOT_FOUND;

    internal.applyQuantumGate(id_slice, nsir_core.identityGate);

    return JAIDE_SUCCESS;
}

pub export fn jaide_version_major() callconv(.C) c_int {
    return 4;
}

pub export fn jaide_version_minor() callconv(.C) c_int {
    return 0;
}

pub export fn jaide_version_patch() callconv(.C) c_int {
    return 0;
}

test "c_api basic graph operations" {
    const testing = std.testing;

    const graph = jaide_create_graph();
    defer jaide_destroy_graph(graph);

    try testing.expect(graph != null);
    try testing.expectEqual(@as(c_int, 0), jaide_graph_node_count(graph));

    const add_result = jaide_add_node(graph, "node1", "test_type");
    try testing.expectEqual(JAIDE_SUCCESS, add_result);
    try testing.expectEqual(@as(c_int, 1), jaide_graph_node_count(graph));

    try testing.expectEqual(@as(c_int, 1), jaide_has_node(graph, "node1"));
    try testing.expectEqual(@as(c_int, 0), jaide_has_node(graph, "nonexistent"));
}

test "c_api quantum operations" {
    const testing = std.testing;

    const graph = jaide_create_graph();
    defer jaide_destroy_graph(graph);

    _ = jaide_add_node(graph, "q1", "qubit");
    _ = jaide_add_node(graph, "q2", "qubit");

    var state: CQuantumState = undefined;
    const get_result = jaide_get_node_quantum_state(graph, "q1", &state);
    try testing.expectEqual(JAIDE_SUCCESS, get_result);

    const set_result = jaide_set_node_quantum_state(graph, "q1", 0.5, 0.5);
    try testing.expectEqual(JAIDE_SUCCESS, set_result);

    _ = jaide_get_node_quantum_state(graph, "q1", &state);
    try testing.expectApproxEqAbs(@as(f64, 0.5), state.real, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.5), state.imag, 0.001);

    const hadamard_result = jaide_apply_hadamard(graph, "q1");
    try testing.expectEqual(JAIDE_SUCCESS, hadamard_result);

    const entangle_result = jaide_entangle_nodes(graph, "q1", "q2");
    try testing.expectEqual(JAIDE_SUCCESS, entangle_result);
}

test "c_api edge operations" {
    const testing = std.testing;

    const graph = jaide_create_graph();
    defer jaide_destroy_graph(graph);

    _ = jaide_add_node(graph, "a", "type_a");
    _ = jaide_add_node(graph, "b", "type_b");

    const edge_result = jaide_add_edge(graph, "a", "b", 0.75);
    try testing.expectEqual(JAIDE_SUCCESS, edge_result);
    try testing.expectEqual(@as(c_int, 1), jaide_graph_edge_count(graph));

    const weight = jaide_get_edge_weight(graph, "a", "b");
    try testing.expectApproxEqAbs(@as(f64, 0.75), weight, 0.001);

    try testing.expectEqual(@as(c_int, 1), jaide_has_edge(graph, "a", "b"));

    const quality_result = jaide_set_edge_quality(graph, "a", "b", 1);
    try testing.expectEqual(JAIDE_SUCCESS, quality_result);
}

test "c_api optimizer" {
    const testing = std.testing;

    const opt = jaide_create_optimizer(100.0, 0.95, 100);
    defer jaide_destroy_optimizer(opt);

    try testing.expect(opt != null);

    const adaptive_result = jaide_set_optimizer_adaptive_cooling(opt, 1);
    try testing.expectEqual(JAIDE_SUCCESS, adaptive_result);

    const min_temp_result = jaide_set_optimizer_min_temperature(opt, 0.01);
    try testing.expectEqual(JAIDE_SUCCESS, min_temp_result);
}
