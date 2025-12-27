const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

pub const EdgeQuality = enum(u8) {
    superposition = 0,
    entangled = 1,
    coherent = 2,
    collapsed = 3,
    fractal = 4,

    pub fn toString(self: EdgeQuality) []const u8 {
        return switch (self) {
            .superposition => "superposition",
            .entangled => "entangled",
            .coherent => "coherent",
            .collapsed => "collapsed",
            .fractal => "fractal",
        };
    }

    pub fn fromString(s: []const u8) ?EdgeQuality {
        if (std.mem.eql(u8, s, "superposition")) return .superposition;
        if (std.mem.eql(u8, s, "entangled")) return .entangled;
        if (std.mem.eql(u8, s, "coherent")) return .coherent;
        if (std.mem.eql(u8, s, "collapsed")) return .collapsed;
        if (std.mem.eql(u8, s, "fractal")) return .fractal;
        return null;
    }
};

pub const Node = struct {
    id: []const u8,
    data: []const u8,
    quantum_state: Complex(f64),
    phase: f64,
    metadata: StringHashMap([]const u8),
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        data: []const u8,
        quantum_real: f64,
        quantum_imag: f64,
        phase: f64,
    ) !Node {
        return Node{
            .id = try allocator.dupe(u8, id),
            .data = try allocator.dupe(u8, data),
            .quantum_state = Complex(f64).init(quantum_real, quantum_imag),
            .phase = phase,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn initWithComplex(
        allocator: Allocator,
        id: []const u8,
        data: []const u8,
        quantum_state: Complex(f64),
        phase: f64,
    ) !Node {
        return Node{
            .id = try allocator.dupe(u8, id),
            .data = try allocator.dupe(u8, data),
            .quantum_state = quantum_state,
            .phase = phase,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Node) void {
        self.allocator.free(self.id);
        self.allocator.free(self.data);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn clone(self: *const Node, allocator: Allocator) !Node {
        var new_node = Node{
            .id = try allocator.dupe(u8, self.id),
            .data = try allocator.dupe(u8, self.data),
            .quantum_state = self.quantum_state,
            .phase = self.phase,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            try new_node.metadata.put(key_copy, val_copy);
        }
        return new_node;
    }

    pub fn setMetadata(self: *Node, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.fetchRemove(key_copy)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
            self.allocator.free(key_copy);
        }
        try self.metadata.put(key_copy, val_copy);
    }

    pub fn getMetadata(self: *const Node, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn magnitude(self: *const Node) f64 {
        return self.quantum_state.magnitude();
    }

    pub fn probability(self: *const Node) f64 {
        const mag = self.magnitude();
        return mag * mag;
    }
};

pub const Edge = struct {
    source: []const u8,
    target: []const u8,
    quality: EdgeQuality,
    weight: f64,
    quantum_correlation: Complex(f64),
    fractal_dimension: f64,
    metadata: StringHashMap([]const u8),
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        source: []const u8,
        target: []const u8,
        quality: EdgeQuality,
        weight: f64,
        quantum_real: f64,
        quantum_imag: f64,
        fractal_dimension: f64,
    ) !Edge {
        return Edge{
            .source = try allocator.dupe(u8, source),
            .target = try allocator.dupe(u8, target),
            .quality = quality,
            .weight = weight,
            .quantum_correlation = Complex(f64).init(quantum_real, quantum_imag),
            .fractal_dimension = fractal_dimension,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn initWithComplex(
        allocator: Allocator,
        source: []const u8,
        target: []const u8,
        quality: EdgeQuality,
        weight: f64,
        quantum_correlation: Complex(f64),
        fractal_dimension: f64,
    ) !Edge {
        return Edge{
            .source = try allocator.dupe(u8, source),
            .target = try allocator.dupe(u8, target),
            .quality = quality,
            .weight = weight,
            .quantum_correlation = quantum_correlation,
            .fractal_dimension = fractal_dimension,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Edge) void {
        self.allocator.free(self.source);
        self.allocator.free(self.target);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn clone(self: *const Edge, allocator: Allocator) !Edge {
        var new_edge = Edge{
            .source = try allocator.dupe(u8, self.source),
            .target = try allocator.dupe(u8, self.target),
            .quality = self.quality,
            .weight = self.weight,
            .quantum_correlation = self.quantum_correlation,
            .fractal_dimension = self.fractal_dimension,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            try new_edge.metadata.put(key_copy, val_copy);
        }
        return new_edge;
    }

    pub fn setMetadata(self: *Edge, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.fetchRemove(key_copy)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
            self.allocator.free(key_copy);
        }
        try self.metadata.put(key_copy, val_copy);
    }

    pub fn getMetadata(self: *const Edge, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn correlationMagnitude(self: *const Edge) f64 {
        return self.quantum_correlation.magnitude();
    }
};

pub const EdgeKey = struct {
    source: []const u8,
    target: []const u8,
};

pub const EdgeKeyContext = struct {
    pub fn hash(self: @This(), key: EdgeKey) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key.source);
        hasher.update(&[_]u8{0});
        hasher.update(key.target);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: EdgeKey, b: EdgeKey) bool {
        _ = self;
        return std.mem.eql(u8, a.source, b.source) and std.mem.eql(u8, a.target, b.target);
    }
};

const StringContext = struct {
    pub fn hash(self: @This(), key: []const u8) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
        _ = self;
        return std.mem.eql(u8, a, b);
    }
};

pub const SelfSimilarRelationalGraph = struct {
    nodes: StringHashMap(Node),
    edges: std.HashMap(EdgeKey, ArrayList(Edge), EdgeKeyContext, std.hash_map.default_max_load_percentage),
    topology_hash: [64]u8,
    fractal_depth: i64,
    quantum_register: StringHashMap(Complex(f64)),
    allocator: Allocator,
    edge_key_allocator: ArrayList([]const u8),

    pub fn init(allocator: Allocator) SelfSimilarRelationalGraph {
        var hash_buf: [64]u8 = undefined;
        @memset(&hash_buf, 0);
        return SelfSimilarRelationalGraph{
            .nodes = StringHashMap(Node).init(allocator),
            .edges = std.HashMap(EdgeKey, ArrayList(Edge), EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .topology_hash = hash_buf,
            .fractal_depth = 0,
            .quantum_register = StringHashMap(Complex(f64)).init(allocator),
            .allocator = allocator,
            .edge_key_allocator = ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *SelfSimilarRelationalGraph) void {
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var node = entry.value_ptr;
            node.deinit();
        }
        self.nodes.deinit();

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            var edge_list = entry.value_ptr;
            for (edge_list.items) |*edge| {
                edge.deinit();
            }
            edge_list.deinit();
        }
        self.edges.deinit();

        for (self.edge_key_allocator.items) |key_str| {
            self.allocator.free(key_str);
        }
        self.edge_key_allocator.deinit();

        var qr_iter = self.quantum_register.iterator();
        while (qr_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.quantum_register.deinit();
    }

    pub fn addNode(self: *SelfSimilarRelationalGraph, node: Node) !void {
        const id_copy = try self.allocator.dupe(u8, node.id);
        errdefer self.allocator.free(id_copy);

        if (self.nodes.contains(node.id)) {
            if (self.nodes.getPtr(node.id)) |existing| {
                existing.deinit();
            }
        }

        try self.nodes.put(id_copy, node);

        const qr_id_copy = try self.allocator.dupe(u8, node.id);
        errdefer self.allocator.free(qr_id_copy);

        if (self.quantum_register.fetchRemove(node.id)) |removed| {
            self.allocator.free(removed.key);
        }
        try self.quantum_register.put(qr_id_copy, node.quantum_state);

        try self.updateTopologyHash();
    }

    pub fn addEdge(self: *SelfSimilarRelationalGraph, edge: Edge) !void {
        const source_copy = try self.allocator.dupe(u8, edge.source);
        errdefer self.allocator.free(source_copy);
        const target_copy = try self.allocator.dupe(u8, edge.target);
        errdefer self.allocator.free(target_copy);

        try self.edge_key_allocator.append(source_copy);
        try self.edge_key_allocator.append(target_copy);

        const key = EdgeKey{ .source = source_copy, .target = target_copy };
        var result = try self.edges.getOrPut(key);
        if (!result.found_existing) {
            result.value_ptr.* = ArrayList(Edge).init(self.allocator);
        }
        try result.value_ptr.append(edge);
        try self.updateTopologyHash();
    }

    pub fn getNode(self: *SelfSimilarRelationalGraph, node_id: []const u8) ?*Node {
        return self.nodes.getPtr(node_id);
    }

    pub fn getNodeConst(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?Node {
        return self.nodes.get(node_id);
    }

    pub fn getEdges(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) ?*ArrayList(Edge) {
        const key = EdgeKey{ .source = source, .target = target };
        return self.edges.getPtr(key);
    }

    pub fn getEdgeList(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) []Edge {
        const key = EdgeKey{ .source = source, .target = target };
        if (self.edges.getPtr(key)) |list| {
            return list.items;
        }
        return &[_]Edge{};
    }

    fn updateTopologyHash(self: *SelfSimilarRelationalGraph) !void {
        var buffer = ArrayList(u8).init(self.allocator);
        defer buffer.deinit();

        var node_keys = ArrayList([]const u8).init(self.allocator);
        defer node_keys.deinit();

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, node_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        for (node_keys.items) |key| {
            if (self.nodes.get(key)) |node| {
                try buffer.appendSlice(key);
                try buffer.append(':');
                try buffer.appendSlice(node.data);

                var state_buf: [64]u8 = undefined;
                const state_str = std.fmt.bufPrint(&state_buf, "{d:.6}+{d:.6}i", .{
                    node.quantum_state.re,
                    node.quantum_state.im,
                }) catch "";
                try buffer.appendSlice(state_str);
                try buffer.append('|');
            }
        }

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            try buffer.appendSlice(key.source);
            try buffer.append('-');
            try buffer.appendSlice(key.target);
            try buffer.append(':');

            for (entry.value_ptr.items) |edge| {
                try buffer.appendSlice(edge.quality.toString());
                var weight_buf: [32]u8 = undefined;
                const weight_str = std.fmt.bufPrint(&weight_buf, "{d:.6}", .{edge.weight}) catch "";
                try buffer.appendSlice(weight_str);
                try buffer.append(',');
            }
            try buffer.append('|');
        }

        var raw_hash: [32]u8 = undefined;
        Sha256.hash(buffer.items, &raw_hash, .{});

        _ = std.fmt.bufPrint(&self.topology_hash, "{s}", .{std.fmt.fmtSliceHexLower(&raw_hash)}) catch {};
    }

    pub fn getTopologyHashHex(self: *const SelfSimilarRelationalGraph) []const u8 {
        var end: usize = 64;
        var hash_idx: usize = 0;
        while (hash_idx < self.topology_hash.len) : (hash_idx += 1) {
            if (self.topology_hash[hash_idx] == 0) {
                end = hash_idx;
                break;
            }
        }
        return self.topology_hash[0..end];
    }

    pub fn calculateFractalDimension(self: *SelfSimilarRelationalGraph) f64 {
        if (self.nodes.count() < 2) {
            return 0.0;
        }

        var total_edges: usize = 0;
        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            total_edges += entry.value_ptr.items.len;
        }

        if (total_edges == 0) {
            return 0.0;
        }

        const box_sizes = [_]usize{ 1, 2, 4, 8, 16 };
        var counts = ArrayList(usize).init(self.allocator);
        defer counts.deinit();

        for (box_sizes) |size| {
            const count = self.boxCount(size);
            if (count > 0) {
                counts.append(count) catch continue;
            }
        }

        if (counts.items.len < 2) {
            return 1.0;
        }

        var sum_x: f64 = 0.0;
        var sum_y: f64 = 0.0;
        var sum_xy: f64 = 0.0;
        var sum_x2: f64 = 0.0;

        var i: usize = 0;
        for (counts.items) |count| {
            if (i >= box_sizes.len) break;
            const size_f = @as(f64, @floatFromInt(box_sizes[i]));
            const count_f = @as(f64, @floatFromInt(count));

            if (size_f > 0 and count_f > 0) {
                const log_size = @log(size_f);
                const log_count = @log(count_f);
                sum_x += log_size;
                sum_y += log_count;
                sum_xy += log_size * log_count;
                sum_x2 += log_size * log_size;
            }
            i += 1;
        }

        const n = @as(f64, @floatFromInt(counts.items.len));
        const denominator = n * sum_x2 - sum_x * sum_x;

        if (@fabs(denominator) < 1e-10) {
            return 1.0;
        }

        const slope = (n * sum_xy - sum_x * sum_y) / denominator;
        return @fabs(slope);
    }

    fn boxCount(self: *SelfSimilarRelationalGraph, box_size: usize) usize {
        if (box_size == 0) return 0;

        var covered_boxes = AutoHashMap(usize, void).init(self.allocator);
        defer covered_boxes.deinit();

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(entry.key_ptr.*);
            const hash_val = hasher.final();
            const box_id = hash_val % box_size;
            covered_boxes.put(box_id, {}) catch continue;
        }

        return covered_boxes.count();
    }

    pub fn applyQuantumGate(
        self: *SelfSimilarRelationalGraph,
        node_id: []const u8,
        gate_fn: *const fn (Complex(f64)) Complex(f64),
    ) void {
        if (self.nodes.getPtr(node_id)) |node| {
            node.quantum_state = gate_fn(node.quantum_state);

            if (self.quantum_register.getPtr(node_id)) |qr_entry| {
                qr_entry.* = node.quantum_state;
            }
        }
    }

    pub fn entangleNodes(self: *SelfSimilarRelationalGraph, node_id1: []const u8, node_id2: []const u8) !void {
        const node1_opt = self.nodes.getPtr(node_id1);
        const node2_opt = self.nodes.getPtr(node_id2);

        if (node1_opt != null and node2_opt != null) {
            const node1 = node1_opt.?;
            const node2 = node2_opt.?;

            const state1 = node1.quantum_state;
            const state2 = node2.quantum_state;
            const sum = state1.add(state2);
            const sqrt2 = Complex(f64).init(@sqrt(2.0), 0.0);
            const entangled_state = sum.div(sqrt2);

            node1.quantum_state = entangled_state;
            node2.quantum_state = entangled_state;

            if (self.quantum_register.getPtr(node_id1)) |qr1| {
                qr1.* = entangled_state;
            }
            if (self.quantum_register.getPtr(node_id2)) |qr2| {
                qr2.* = entangled_state;
            }

            const fractal_dim = self.calculateFractalDimension();
            const edge = try Edge.initWithComplex(
                self.allocator,
                node_id1,
                node_id2,
                .entangled,
                1.0,
                entangled_state,
                fractal_dim,
            );
            try self.addEdge(edge);
        }
    }

    pub fn measure(self: *SelfSimilarRelationalGraph, node_id: []const u8) f64 {
        const node_opt = self.nodes.getPtr(node_id);
        if (node_opt == null) {
            return 0.0;
        }

        const node = node_opt.?;
        const state = node.quantum_state;
        const mag = state.magnitude();
        const prob = mag * mag;

        const collapsed_value: f64 = if (prob > 0.5) 1.0 else 0.0;
        node.quantum_state = Complex(f64).init(collapsed_value, 0.0);

        if (self.quantum_register.getPtr(node_id)) |qr| {
            qr.* = node.quantum_state;
        }

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key.source, node_id) or std.mem.eql(u8, key.target, node_id)) {
                for (entry.value_ptr.items) |*edge| {
                    if (edge.quality == .entangled) {
                        edge.quality = .collapsed;
                    }
                }
            }
        }

        return prob;
    }

    pub fn propagateInformation(
        self: *SelfSimilarRelationalGraph,
        source_id: []const u8,
        depth: i64,
    ) !ArrayList([]const u8) {
        var visited = std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer {
            var iter = visited.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            visited.deinit();
        }

        const QueueItem = struct { id: []const u8, d: i64 };
        var queue = ArrayList(QueueItem).init(self.allocator);
        defer {
            for (queue.items) |item| {
                self.allocator.free(item.id);
            }
            queue.deinit();
        }

        try queue.append(.{ .id = try self.allocator.dupe(u8, source_id), .d = 0 });

        while (queue.items.len > 0) {
            const current = queue.orderedRemove(0);
            defer self.allocator.free(current.id);

            if (visited.contains(current.id) or current.d > depth) {
                continue;
            }

            const id_copy = try self.allocator.dupe(u8, current.id);
            try visited.put(id_copy, {});

            var edge_iter = self.edges.iterator();
            while (edge_iter.next()) |entry| {
                const key = entry.key_ptr.*;
                if (std.mem.eql(u8, key.source, current.id)) {
                    for (entry.value_ptr.items) |edge| {
                        if (edge.quality != .collapsed) {
                            const next_id = try self.allocator.dupe(u8, key.target);
                            try queue.append(.{ .id = next_id, .d = current.d + 1 });
                        }
                    }
                } else if (std.mem.eql(u8, key.target, current.id)) {
                    for (entry.value_ptr.items) |edge| {
                        if (edge.quality != .collapsed) {
                            const next_id = try self.allocator.dupe(u8, key.source);
                            try queue.append(.{ .id = next_id, .d = current.d + 1 });
                        }
                    }
                }
            }
        }

        var result = ArrayList([]const u8).init(self.allocator);
        var iter = visited.iterator();
        while (iter.next()) |entry| {
            const id_copy = try self.allocator.dupe(u8, entry.key_ptr.*);
            try result.append(id_copy);
        }
        return result;
    }

    pub fn selfSimilarityCheck(
        self: *SelfSimilarRelationalGraph,
        subgraph_nodes: std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage),
    ) f64 {
        const subgraph_count = subgraph_nodes.count();
        const total_nodes = self.nodes.count();

        if (subgraph_count == 0 or subgraph_count == total_nodes) {
            return 0.0;
        }

        var subgraph_edges: usize = 0;
        const total_subgraph_possible = subgraph_count * (subgraph_count -| 1);

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            if (subgraph_nodes.contains(key.source) and subgraph_nodes.contains(key.target)) {
                subgraph_edges += entry.value_ptr.items.len;
            }
        }

        var total_edges: usize = 0;
        var all_edge_iter = self.edges.iterator();
        while (all_edge_iter.next()) |entry| {
            total_edges += entry.value_ptr.items.len;
        }

        const total_possible = total_nodes * (total_nodes -| 1);

        if (total_possible == 0 or total_subgraph_possible == 0) {
            return 0.0;
        }

        const graph_density = @as(f64, @floatFromInt(total_edges)) / @as(f64, @floatFromInt(total_possible));
        const subgraph_density = @as(f64, @floatFromInt(subgraph_edges)) / @as(f64, @floatFromInt(total_subgraph_possible));

        return 1.0 - @fabs(graph_density - subgraph_density);
    }

    pub fn selfSimilarityCheckSimple(self: *SelfSimilarRelationalGraph, subgraph_node_ids: []const []const u8) f64 {
        var subgraph_nodes = std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer subgraph_nodes.deinit();

        for (subgraph_node_ids) |id| {
            subgraph_nodes.put(id, {}) catch continue;
        }

        return self.selfSimilarityCheck(subgraph_nodes);
    }

    pub fn encodeInformation(self: *SelfSimilarRelationalGraph, data: []const u8) ![]const u8 {
        var raw_hash: [32]u8 = undefined;
        Sha256.hash(data, &raw_hash, .{});

        var node_id_buf: [16]u8 = undefined;
        _ = std.fmt.bufPrint(&node_id_buf, "{s}", .{std.fmt.fmtSliceHexLower(raw_hash[0..8])}) catch {};
        const node_id = try self.allocator.dupe(u8, &node_id_buf);

        const len_f = @as(f64, @floatFromInt(data.len));
        const quantum_state = Complex(f64).init(
            @cos(len_f * 0.1),
            @sin(len_f * 0.1),
        );

        var node = try Node.initWithComplex(
            self.allocator,
            node_id,
            data,
            quantum_state,
            len_f * 0.1,
        );

        var encoding_time_buf: [32]u8 = undefined;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(data);
        const time_hash = hasher.final();
        const time_str = std.fmt.bufPrint(&encoding_time_buf, "{d}", .{time_hash}) catch "";
        try node.setMetadata("encoding_time", time_str);

        try self.addNode(node);

        if (self.nodes.count() > 1) {
            var prev_nodes = ArrayList([]const u8).init(self.allocator);
            defer prev_nodes.deinit();

            var node_iter = self.nodes.iterator();
            while (node_iter.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.*, node_id)) {
                    try prev_nodes.append(entry.key_ptr.*);
                }
            }

            const start_idx = if (prev_nodes.items.len > 3) prev_nodes.items.len - 3 else 0;
            for (prev_nodes.items[start_idx..]) |prev_id| {
                const correlation = self.calculateCorrelation(node_id, prev_id);
                const magnitude = correlation.magnitude();
                const quality: EdgeQuality = if (magnitude > 0.5) .coherent else .superposition;

                const edge = try Edge.initWithComplex(
                    self.allocator,
                    prev_id,
                    node_id,
                    quality,
                    magnitude,
                    correlation,
                    self.calculateFractalDimension(),
                );
                try self.addEdge(edge);
            }
        }

        return node_id;
    }

    pub fn decodeInformation(self: *SelfSimilarRelationalGraph, node_id: []const u8) ?[]const u8 {
        const node = self.getNode(node_id);
        if (node) |n| {
            return n.data;
        }
        return null;
    }

    fn calculateCorrelation(self: *SelfSimilarRelationalGraph, node_id1: []const u8, node_id2: []const u8) Complex(f64) {
        const node1_opt = self.nodes.get(node_id1);
        const node2_opt = self.nodes.get(node_id2);

        if (node1_opt == null or node2_opt == null) {
            return Complex(f64).init(0.0, 0.0);
        }

        const state1 = node1_opt.?.quantum_state;
        const state2_conj = node2_opt.?.quantum_state.conjugate();

        return state1.mul(state2_conj);
    }

    pub fn nodeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.nodes.count();
    }

    pub fn edgeCount(self: *const SelfSimilarRelationalGraph) usize {
        var count: usize = 0;
        var iter = self.edges.iterator();
        while (iter.next()) |entry| {
            count += entry.value_ptr.items.len;
        }
        return count;
    }

    pub fn clear(self: *SelfSimilarRelationalGraph) void {
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var node = entry.value_ptr;
            node.deinit();
        }
        self.nodes.clearRetainingCapacity();

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            var edge_list = entry.value_ptr;
            for (edge_list.items) |*edge| {
                edge.deinit();
            }
            edge_list.deinit();
        }
        self.edges.clearRetainingCapacity();

        for (self.edge_key_allocator.items) |key_str| {
            self.allocator.free(key_str);
        }
        self.edge_key_allocator.clearRetainingCapacity();

        var qr_iter = self.quantum_register.iterator();
        while (qr_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.quantum_register.clearRetainingCapacity();

        @memset(&self.topology_hash, 0);
        self.fractal_depth = 0;
    }

    pub fn getQuantumState(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?Complex(f64) {
        return self.quantum_register.get(node_id);
    }

    pub fn setQuantumState(self: *SelfSimilarRelationalGraph, node_id: []const u8, state: Complex(f64)) !void {
        if (self.nodes.getPtr(node_id)) |node| {
            node.quantum_state = state;
        }
        if (self.quantum_register.getPtr(node_id)) |qr| {
            qr.* = state;
        } else {
            const id_copy = try self.allocator.dupe(u8, node_id);
            try self.quantum_register.put(id_copy, state);
        }
    }

    pub fn getAllNodeIds(self: *const SelfSimilarRelationalGraph, allocator: Allocator) !ArrayList([]const u8) {
        var result = ArrayList([]const u8).init(allocator);
        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            try result.append(entry.key_ptr.*);
        }
        return result;
    }

    pub fn hasNode(self: *const SelfSimilarRelationalGraph, node_id: []const u8) bool {
        return self.nodes.contains(node_id);
    }

    pub fn hasEdge(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) bool {
        const key = EdgeKey{ .source = source, .target = target };
        return self.edges.contains(key);
    }
};

pub fn hadamardGate(state: Complex(f64)) Complex(f64) {
    const sqrt2_inv = 1.0 / @sqrt(2.0);
    const h = Complex(f64).init(sqrt2_inv, 0.0);
    return state.mul(h);
}

pub fn pauliXGate(state: Complex(f64)) Complex(f64) {
    return Complex(f64).init(state.im, state.re);
}

pub fn pauliYGate(state: Complex(f64)) Complex(f64) {
    return Complex(f64).init(state.im, -state.re);
}

pub fn pauliZGate(state: Complex(f64)) Complex(f64) {
    return Complex(f64).init(state.re, -state.im);
}

pub fn phaseGate(phase: f64) fn (Complex(f64)) Complex(f64) {
    _ = phase;
    return struct {
        fn apply(state: Complex(f64)) Complex(f64) {
            return state;
        }
    }.apply;
}

pub fn identityGate(state: Complex(f64)) Complex(f64) {
    return state;
}

test "EdgeQuality toString" {
    const testing = std.testing;
    try testing.expectEqualStrings("superposition", EdgeQuality.superposition.toString());
    try testing.expectEqualStrings("entangled", EdgeQuality.entangled.toString());
    try testing.expectEqualStrings("coherent", EdgeQuality.coherent.toString());
    try testing.expectEqualStrings("collapsed", EdgeQuality.collapsed.toString());
    try testing.expectEqualStrings("fractal", EdgeQuality.fractal.toString());
}

test "EdgeQuality fromString" {
    const testing = std.testing;
    try testing.expectEqual(EdgeQuality.superposition, EdgeQuality.fromString("superposition").?);
    try testing.expectEqual(EdgeQuality.entangled, EdgeQuality.fromString("entangled").?);
    try testing.expect(EdgeQuality.fromString("invalid") == null);
}

test "Node init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var node = try Node.init(allocator, "node1", "test data", 1.0, 0.5, 0.25);
    defer node.deinit();

    try testing.expectEqualStrings("node1", node.id);
    try testing.expectEqualStrings("test data", node.data);
    try testing.expectApproxEqAbs(@as(f64, 1.0), node.quantum_state.re, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.5), node.quantum_state.im, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.25), node.phase, 0.001);
}

test "Edge init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var edge = try Edge.init(allocator, "src", "tgt", .coherent, 0.75, 1.0, 0.5, 1.5);
    defer edge.deinit();

    try testing.expectEqualStrings("src", edge.source);
    try testing.expectEqualStrings("tgt", edge.target);
    try testing.expectEqual(EdgeQuality.coherent, edge.quality);
    try testing.expectApproxEqAbs(@as(f64, 0.75), edge.weight, 0.001);
}

test "SelfSimilarRelationalGraph basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var node1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(node1);

    var node2 = try Node.init(allocator, "n2", "data2", 0.0, 1.0, 0.5);
    try graph.addNode(node2);

    try testing.expectEqual(@as(usize, 2), graph.nodeCount());

    const retrieved = graph.getNode("n1");
    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("data1", retrieved.?.data);
}

test "SelfSimilarRelationalGraph edge operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var node1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(node1);

    var node2 = try Node.init(allocator, "n2", "data2", 0.0, 1.0, 0.5);
    try graph.addNode(node2);

    var edge = try Edge.init(allocator, "n1", "n2", .coherent, 0.8, 0.5, 0.5, 1.2);
    try graph.addEdge(edge);

    try testing.expectEqual(@as(usize, 1), graph.edgeCount());
}

test "SelfSimilarRelationalGraph measure" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var node = try Node.init(allocator, "n1", "data", 0.8, 0.0, 0.0);
    try graph.addNode(node);

    const prob = graph.measure("n1");
    try testing.expect(prob >= 0.0 and prob <= 1.0);

    const measured_node = graph.getNode("n1");
    try testing.expect(measured_node != null);
    const state = measured_node.?.quantum_state;
    try testing.expect(state.re == 1.0 or state.re == 0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), state.im, 0.001);
}

test "SelfSimilarRelationalGraph entangle nodes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var node1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(node1);

    var node2 = try Node.init(allocator, "n2", "data2", 0.0, 1.0, 0.0);
    try graph.addNode(node2);

    try graph.entangleNodes("n1", "n2");

    try testing.expectEqual(@as(usize, 1), graph.edgeCount());

    const n1 = graph.getNode("n1");
    const n2 = graph.getNode("n2");
    try testing.expect(n1 != null and n2 != null);

    try testing.expectApproxEqAbs(n1.?.quantum_state.re, n2.?.quantum_state.re, 0.001);
    try testing.expectApproxEqAbs(n1.?.quantum_state.im, n2.?.quantum_state.im, 0.001);
}

test "SelfSimilarRelationalGraph encode and decode" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const node_id = try graph.encodeInformation("hello world");
    defer allocator.free(node_id);

    try testing.expectEqual(@as(usize, 1), graph.nodeCount());

    const decoded = graph.decodeInformation(node_id);
    try testing.expect(decoded != null);
    try testing.expectEqualStrings("hello world", decoded.?);
}

test "SelfSimilarRelationalGraph propagate information" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "d1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "d2", 1.0, 0.0, 0.0);
    try graph.addNode(n2);
    var n3 = try Node.init(allocator, "n3", "d3", 1.0, 0.0, 0.0);
    try graph.addNode(n3);

    var e1 = try Edge.init(allocator, "n1", "n2", .coherent, 1.0, 1.0, 0.0, 1.0);
    try graph.addEdge(e1);
    var e2 = try Edge.init(allocator, "n2", "n3", .coherent, 1.0, 1.0, 0.0, 1.0);
    try graph.addEdge(e2);

    var visited = try graph.propagateInformation("n1", 2);
    defer {
        for (visited.items) |id| {
            allocator.free(id);
        }
        visited.deinit();
    }

    try testing.expect(visited.items.len >= 1);
}

test "quantum gates" {
    const testing = std.testing;

    const state = Complex(f64).init(1.0, 0.0);

    const h_result = hadamardGate(state);
    try testing.expect(@fabs(h_result.magnitude() - 1.0 / @sqrt(2.0)) < 0.001);

    const x_result = pauliXGate(Complex(f64).init(0.5, 0.3));
    try testing.expectApproxEqAbs(@as(f64, 0.3), x_result.re, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.5), x_result.im, 0.001);

    const id_result = identityGate(state);
    try testing.expectApproxEqAbs(state.re, id_result.re, 0.001);
    try testing.expectApproxEqAbs(state.im, id_result.im, 0.001);
}
