const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const chaos_core = @import("chaos_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const ChaosCoreKernel = chaos_core.ChaosCoreKernel;

pub const ExtractionStage = enum(u8) {
    tokenization = 0,
    triplet_extraction = 1,
    validation = 2,
    integration = 3,
    indexing = 4,

    pub fn toString(self: ExtractionStage) []const u8 {
        return switch (self) {
            .tokenization => "tokenization",
            .triplet_extraction => "triplet_extraction",
            .validation => "validation",
            .integration => "integration",
            .indexing => "indexing",
        };
    }

    pub fn fromString(s: []const u8) ?ExtractionStage {
        if (std.mem.eql(u8, s, "tokenization")) return .tokenization;
        if (std.mem.eql(u8, s, "triplet_extraction")) return .triplet_extraction;
        if (std.mem.eql(u8, s, "validation")) return .validation;
        if (std.mem.eql(u8, s, "integration")) return .integration;
        if (std.mem.eql(u8, s, "indexing")) return .indexing;
        return null;
    }

    pub fn next(self: ExtractionStage) ?ExtractionStage {
        return switch (self) {
            .tokenization => .triplet_extraction,
            .triplet_extraction => .validation,
            .validation => .integration,
            .integration => .indexing,
            .indexing => null,
        };
    }
};

pub const RelationalTriplet = struct {
    subject: []const u8,
    relation: []const u8,
    object: []const u8,
    confidence: f64,
    source_hash: [32]u8,
    extraction_time: i64,
    allocator: Allocator,
    metadata: StringHashMap([]const u8),

    pub fn init(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence: f64,
    ) !RelationalTriplet {
        var source_hash: [32]u8 = undefined;
        var buffer = ArrayList(u8).init(allocator);
        defer buffer.deinit();
        try buffer.appendSlice(subject);
        try buffer.append(':');
        try buffer.appendSlice(relation);
        try buffer.append(':');
        try buffer.appendSlice(object);
        Sha256.hash(buffer.items, &source_hash, .{});

        return RelationalTriplet{
            .subject = try allocator.dupe(u8, subject),
            .relation = try allocator.dupe(u8, relation),
            .object = try allocator.dupe(u8, object),
            .confidence = confidence,
            .source_hash = source_hash,
            .extraction_time = std.time.milliTimestamp(),
            .allocator = allocator,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn initWithHash(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence: f64,
        source_hash: [32]u8,
        extraction_time: i64,
    ) !RelationalTriplet {
        return RelationalTriplet{
            .subject = try allocator.dupe(u8, subject),
            .relation = try allocator.dupe(u8, relation),
            .object = try allocator.dupe(u8, object),
            .confidence = confidence,
            .source_hash = source_hash,
            .extraction_time = extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *RelationalTriplet) void {
        self.allocator.free(self.subject);
        self.allocator.free(self.relation);
        self.allocator.free(self.object);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn clone(self: *const RelationalTriplet, allocator: Allocator) !RelationalTriplet {
        var new_triplet = RelationalTriplet{
            .subject = try allocator.dupe(u8, self.subject),
            .relation = try allocator.dupe(u8, self.relation),
            .object = try allocator.dupe(u8, self.object),
            .confidence = self.confidence,
            .source_hash = self.source_hash,
            .extraction_time = self.extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            try new_triplet.metadata.put(key_copy, val_copy);
        }
        return new_triplet;
    }

    pub fn computeHash(self: *const RelationalTriplet) [32]u8 {
        var buffer: [1024]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const writer = stream.writer();
        writer.print("{s}:{s}:{s}:{d:.6}:{d}", .{
            self.subject,
            self.relation,
            self.object,
            self.confidence,
            self.extraction_time,
        }) catch {};
        var hash: [32]u8 = undefined;
        Sha256.hash(stream.getWritten(), &hash, .{});
        return hash;
    }

    pub fn setMetadata(self: *RelationalTriplet, key: []const u8, value: []const u8) !void {
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

    pub fn getMetadata(self: *const RelationalTriplet, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn equals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, self.subject, other.subject) and
            std.mem.eql(u8, self.relation, other.relation) and
            std.mem.eql(u8, self.object, other.object);
    }

    pub fn hashEquals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, &self.source_hash, &other.source_hash);
    }

    pub fn toGraphElements(self: *const RelationalTriplet, allocator: Allocator) !struct {
        subject_node: Node,
        object_node: Node,
        edge: Edge,
    } {
        var subject_id_hash: [32]u8 = undefined;
        Sha256.hash(self.subject, &subject_id_hash, .{});
        var subject_id: [16]u8 = undefined;
        @memcpy(&subject_id, subject_id_hash[0..16]);

        var object_id_hash: [32]u8 = undefined;
        Sha256.hash(self.object, &object_id_hash, .{});
        var object_id: [16]u8 = undefined;
        @memcpy(&object_id, object_id_hash[0..16]);

        var subject_id_str: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&subject_id_str, "{s}", .{std.fmt.fmtSliceHexLower(&subject_id)}) catch {};
        var object_id_str: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&object_id_str, "{s}", .{std.fmt.fmtSliceHexLower(&object_id)}) catch {};

        const phase = @as(f64, @floatFromInt(@mod(self.extraction_time, 360))) / 360.0 * std.math.pi * 2.0;
        const quantum_state = Complex(f64).init(self.confidence, @sqrt(1.0 - self.confidence * self.confidence));

        var subject_node = try Node.initWithComplex(
            allocator,
            &subject_id_str,
            self.subject,
            quantum_state,
            phase,
        );
        try subject_node.setMetadata("type", "entity");
        try subject_node.setMetadata("role", "subject");

        var object_node = try Node.initWithComplex(
            allocator,
            &object_id_str,
            self.object,
            quantum_state,
            phase,
        );
        try object_node.setMetadata("type", "entity");
        try object_node.setMetadata("role", "object");

        var edge = try Edge.initWithComplex(
            allocator,
            &subject_id_str,
            &object_id_str,
            .coherent,
            self.confidence,
            quantum_state,
            1.0,
        );
        try edge.setMetadata("relation", self.relation);

        var conf_buf: [32]u8 = undefined;
        const conf_str = std.fmt.bufPrint(&conf_buf, "{d:.6}", .{self.confidence}) catch "0.0";
        try edge.setMetadata("confidence", conf_str);

        return .{
            .subject_node = subject_node,
            .object_node = object_node,
            .edge = edge,
        };
    }
};

pub const ValidationResult = struct {
    triplet: *RelationalTriplet,
    is_valid: bool,
    confidence_adjusted: f64,
    validation_method: []const u8,
    conflicts: ArrayList(*RelationalTriplet),
    anomaly_score: f64,
    validation_time: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, triplet: *RelationalTriplet) ValidationResult {
        return ValidationResult{
            .triplet = triplet,
            .is_valid = true,
            .confidence_adjusted = triplet.confidence,
            .validation_method = "",
            .conflicts = ArrayList(*RelationalTriplet).init(allocator),
            .anomaly_score = 0.0,
            .validation_time = std.time.milliTimestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        self.conflicts.deinit();
    }

    pub fn addConflict(self: *ValidationResult, conflict: *RelationalTriplet) !void {
        try self.conflicts.append(conflict);
    }

    pub fn hasConflicts(self: *const ValidationResult) bool {
        return self.conflicts.items.len > 0;
    }

    pub fn conflictCount(self: *const ValidationResult) usize {
        return self.conflicts.items.len;
    }

    pub fn setValidationMethod(self: *ValidationResult, method: []const u8) void {
        self.validation_method = method;
    }
};

const StringContext = struct {
    pub fn hash(_: @This(), key: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
};

pub const TripletIndex = std.HashMap([]const u8, ArrayList(*RelationalTriplet), StringContext, std.hash_map.default_max_load_percentage);

pub const KnowledgeGraphIndex = struct {
    subject_index: TripletIndex,
    relation_index: TripletIndex,
    object_index: TripletIndex,
    all_triplets: ArrayList(*RelationalTriplet),
    allocator: Allocator,
    triplet_count: usize,

    pub fn init(allocator: Allocator) KnowledgeGraphIndex {
        return KnowledgeGraphIndex{
            .subject_index = TripletIndex.init(allocator),
            .relation_index = TripletIndex.init(allocator),
            .object_index = TripletIndex.init(allocator),
            .all_triplets = ArrayList(*RelationalTriplet).init(allocator),
            .allocator = allocator,
            .triplet_count = 0,
        };
    }

    pub fn deinit(self: *KnowledgeGraphIndex) void {
        var subject_iter = self.subject_index.iterator();
        while (subject_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.subject_index.deinit();

        var relation_iter = self.relation_index.iterator();
        while (relation_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.relation_index.deinit();

        var object_iter = self.object_index.iterator();
        while (object_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.object_index.deinit();

        for (self.all_triplets.items) |triplet| {
            triplet.deinit();
            self.allocator.destroy(triplet);
        }
        self.all_triplets.deinit();
    }

    pub fn index(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) !void {
        try self.all_triplets.append(triplet);

        var subject_result = try self.subject_index.getOrPut(triplet.subject);
        if (!subject_result.found_existing) {
            subject_result.key_ptr.* = try self.allocator.dupe(u8, triplet.subject);
            subject_result.value_ptr.* = ArrayList(*RelationalTriplet).init(self.allocator);
        }
        try subject_result.value_ptr.append(triplet);

        var relation_result = try self.relation_index.getOrPut(triplet.relation);
        if (!relation_result.found_existing) {
            relation_result.key_ptr.* = try self.allocator.dupe(u8, triplet.relation);
            relation_result.value_ptr.* = ArrayList(*RelationalTriplet).init(self.allocator);
        }
        try relation_result.value_ptr.append(triplet);

        var object_result = try self.object_index.getOrPut(triplet.object);
        if (!object_result.found_existing) {
            object_result.key_ptr.* = try self.allocator.dupe(u8, triplet.object);
            object_result.value_ptr.* = ArrayList(*RelationalTriplet).init(self.allocator);
        }
        try object_result.value_ptr.append(triplet);

        self.triplet_count += 1;
    }

    pub fn query(
        self: *KnowledgeGraphIndex,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
        allocator: Allocator,
    ) !ArrayList(*RelationalTriplet) {
        var results = ArrayList(*RelationalTriplet).init(allocator);

        if (subject == null and relation == null and object == null) {
            for (self.all_triplets.items) |triplet| {
                try results.append(triplet);
            }
            return results;
        }

        var candidates: ?*ArrayList(*RelationalTriplet) = null;

        if (subject) |s| {
            if (self.subject_index.getPtr(s)) |list| {
                candidates = list;
            }
        }

        if (candidates == null) {
            if (relation) |r| {
                if (self.relation_index.getPtr(r)) |list| {
                    candidates = list;
                }
            }
        }

        if (candidates == null) {
            if (object) |o| {
                if (self.object_index.getPtr(o)) |list| {
                    candidates = list;
                }
            }
        }

        if (candidates) |cands| {
            for (cands.items) |triplet| {
                var matches = true;

                if (subject) |s| {
                    if (!std.mem.eql(u8, triplet.subject, s)) {
                        matches = false;
                    }
                }
                if (matches and relation != null) {
                    if (!std.mem.eql(u8, triplet.relation, relation.?)) {
                        matches = false;
                    }
                }
                if (matches and object != null) {
                    if (!std.mem.eql(u8, triplet.object, object.?)) {
                        matches = false;
                    }
                }

                if (matches) {
                    try results.append(triplet);
                }
            }
        }

        return results;
    }

    pub fn queryBySubject(self: *KnowledgeGraphIndex, subject: []const u8) []*RelationalTriplet {
        if (self.subject_index.getPtr(subject)) |list| {
            return list.items;
        }
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByRelation(self: *KnowledgeGraphIndex, relation: []const u8) []*RelationalTriplet {
        if (self.relation_index.getPtr(relation)) |list| {
            return list.items;
        }
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByObject(self: *KnowledgeGraphIndex, object: []const u8) []*RelationalTriplet {
        if (self.object_index.getPtr(object)) |list| {
            return list.items;
        }
        return &[_]*RelationalTriplet{};
    }

    pub fn remove(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) bool {
        var removed = false;

        if (self.subject_index.getPtr(triplet.subject)) |list| {
            var i: usize = 0;
            while (i < list.items.len) {
                if (list.items[i] == triplet) {
                    _ = list.orderedRemove(i);
                    removed = true;
                } else {
                    i += 1;
                }
            }
        }

        if (self.relation_index.getPtr(triplet.relation)) |list| {
            var i: usize = 0;
            while (i < list.items.len) {
                if (list.items[i] == triplet) {
                    _ = list.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }

        if (self.object_index.getPtr(triplet.object)) |list| {
            var i: usize = 0;
            while (i < list.items.len) {
                if (list.items[i] == triplet) {
                    _ = list.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }

        var idx: usize = 0;
        while (idx < self.all_triplets.items.len) {
            if (self.all_triplets.items[idx] == triplet) {
                _ = self.all_triplets.orderedRemove(idx);
            } else {
                idx += 1;
            }
        }

        if (removed) {
            self.triplet_count -= 1;
        }

        return removed;
    }

    pub fn count(self: *const KnowledgeGraphIndex) usize {
        return self.triplet_count;
    }

    pub fn getUniqueSubjects(self: *const KnowledgeGraphIndex) usize {
        return self.subject_index.count();
    }

    pub fn getUniqueRelations(self: *const KnowledgeGraphIndex) usize {
        return self.relation_index.count();
    }

    pub fn getUniqueObjects(self: *const KnowledgeGraphIndex) usize {
        return self.object_index.count();
    }
};

pub const StreamBuffer = struct {
    buffer: []?*RelationalTriplet,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
    allocator: Allocator,
    overflow_count: usize,
    total_pushed: usize,
    total_popped: usize,

    pub fn init(allocator: Allocator, capacity: usize) !StreamBuffer {
        const buffer = try allocator.alloc(?*RelationalTriplet, capacity);
        @memset(buffer, null);
        return StreamBuffer{
            .buffer = buffer,
            .capacity = capacity,
            .head = 0,
            .tail = 0,
            .size = 0,
            .allocator = allocator,
            .overflow_count = 0,
            .total_pushed = 0,
            .total_popped = 0,
        };
    }

    pub fn deinit(self: *StreamBuffer) void {
        for (self.buffer) |item| {
            if (item) |triplet| {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }
        self.allocator.free(self.buffer);
    }

    pub fn push(self: *StreamBuffer, triplet: *RelationalTriplet) !bool {
        if (self.isFull()) {
            self.overflow_count += 1;
            return false;
        }

        self.buffer[self.tail] = triplet;
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
        self.total_pushed += 1;
        return true;
    }

    pub fn pop(self: *StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) {
            return null;
        }

        const triplet = self.buffer[self.head];
        self.buffer[self.head] = null;
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        self.total_popped += 1;
        return triplet;
    }

    pub fn peek(self: *const StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) {
            return null;
        }
        return self.buffer[self.head];
    }

    pub fn peekAt(self: *const StreamBuffer, offset: usize) ?*RelationalTriplet {
        if (offset >= self.size) {
            return null;
        }
        const idx = (self.head + offset) % self.capacity;
        return self.buffer[idx];
    }

    pub fn isFull(self: *const StreamBuffer) bool {
        return self.size >= self.capacity;
    }

    pub fn isEmpty(self: *const StreamBuffer) bool {
        return self.size == 0;
    }

    pub fn getSize(self: *const StreamBuffer) usize {
        return self.size;
    }

    pub fn getCapacity(self: *const StreamBuffer) usize {
        return self.capacity;
    }

    pub fn clear(self: *StreamBuffer) void {
        while (!self.isEmpty()) {
            if (self.pop()) |triplet| {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }
        self.head = 0;
        self.tail = 0;
    }

    pub fn getUtilization(self: *const StreamBuffer) f64 {
        if (self.capacity == 0) return 0.0;
        return @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity));
    }
};

pub const PipelineResult = struct {
    triplets_extracted: usize,
    triplets_validated: usize,
    triplets_integrated: usize,
    conflicts_resolved: usize,
    processing_time_ns: i64,
    stage: ExtractionStage,
    success: bool,
    error_message: ?[]const u8,

    pub fn init() PipelineResult {
        return PipelineResult{
            .triplets_extracted = 0,
            .triplets_validated = 0,
            .triplets_integrated = 0,
            .conflicts_resolved = 0,
            .processing_time_ns = 0,
            .stage = .tokenization,
            .success = true,
            .error_message = null,
        };
    }

    pub fn merge(self: *PipelineResult, other: PipelineResult) void {
        self.triplets_extracted += other.triplets_extracted;
        self.triplets_validated += other.triplets_validated;
        self.triplets_integrated += other.triplets_integrated;
        self.conflicts_resolved += other.conflicts_resolved;
        self.processing_time_ns += other.processing_time_ns;
    }
};

pub const PipelineStatistics = struct {
    total_extractions: usize,
    total_validations: usize,
    total_integrations: usize,
    average_confidence: f64,
    conflict_rate: f64,
    throughput: f64,
    buffer_utilization: f64,
    unique_subjects: usize,
    unique_relations: usize,
    unique_objects: usize,
    uptime_ms: i64,

    pub fn init() PipelineStatistics {
        return PipelineStatistics{
            .total_extractions = 0,
            .total_validations = 0,
            .total_integrations = 0,
            .average_confidence = 0.0,
            .conflict_rate = 0.0,
            .throughput = 0.0,
            .buffer_utilization = 0.0,
            .unique_subjects = 0,
            .unique_relations = 0,
            .unique_objects = 0,
            .uptime_ms = 0,
        };
    }
};

pub const RelationPattern = struct {
    pattern: []const u8,
    relation_type: []const u8,
    weight: f64,
};

pub const TokenizerConfig = struct {
    min_entity_length: usize,
    max_entity_length: usize,
    min_confidence_threshold: f64,
    enable_coreference: bool,
    language: []const u8,

    pub fn default() TokenizerConfig {
        return TokenizerConfig{
            .min_entity_length = 2,
            .max_entity_length = 100,
            .min_confidence_threshold = 0.3,
            .enable_coreference = true,
            .language = "en",
        };
    }
};

pub const CREVPipeline = struct {
    kernel: *ChaosCoreKernel,
    triplet_buffer: StreamBuffer,
    knowledge_index: KnowledgeGraphIndex,
    validation_threshold: f64,
    extraction_count: usize,
    validation_count: usize,
    integration_count: usize,
    conflict_count: usize,
    allocator: Allocator,
    start_time: i64,
    total_confidence_sum: f64,
    relation_patterns: ArrayList(RelationPattern),
    tokenizer_config: TokenizerConfig,
    relation_statistics: StringHashMap(RelationStatistics),
    entity_statistics: StringHashMap(EntityStatistics),
    is_running: bool,

    pub const RelationStatistics = struct {
        count: usize,
        total_confidence: f64,
        variance_sum: f64,
        avg_confidence: f64,

        pub fn init() RelationStatistics {
            return RelationStatistics{
                .count = 0,
                .total_confidence = 0.0,
                .variance_sum = 0.0,
                .avg_confidence = 0.0,
            };
        }

        pub fn update(self: *RelationStatistics, confidence: f64) void {
            const old_avg = self.avg_confidence;
            self.count += 1;
            self.total_confidence += confidence;
            self.avg_confidence = self.total_confidence / @as(f64, @floatFromInt(self.count));
            if (self.count > 1) {
                self.variance_sum += (confidence - old_avg) * (confidence - self.avg_confidence);
            }
        }

        pub fn getVariance(self: *const RelationStatistics) f64 {
            if (self.count < 2) return 0.0;
            return self.variance_sum / @as(f64, @floatFromInt(self.count - 1));
        }

        pub fn getStdDev(self: *const RelationStatistics) f64 {
            return @sqrt(self.getVariance());
        }
    };

    pub const EntityStatistics = struct {
        count: usize,
        as_subject: usize,
        as_object: usize,
        total_confidence: f64,

        pub fn init() EntityStatistics {
            return EntityStatistics{
                .count = 0,
                .as_subject = 0,
                .as_object = 0,
                .total_confidence = 0.0,
            };
        }
    };

    pub fn init(allocator: Allocator, kernel: *ChaosCoreKernel) !CREVPipeline {
        var pipeline = CREVPipeline{
            .kernel = kernel,
            .triplet_buffer = try StreamBuffer.init(allocator, 10000),
            .knowledge_index = KnowledgeGraphIndex.init(allocator),
            .validation_threshold = 0.5,
            .extraction_count = 0,
            .validation_count = 0,
            .integration_count = 0,
            .conflict_count = 0,
            .allocator = allocator,
            .start_time = std.time.milliTimestamp(),
            .total_confidence_sum = 0.0,
            .relation_patterns = ArrayList(RelationPattern).init(allocator),
            .tokenizer_config = TokenizerConfig.default(),
            .relation_statistics = StringHashMap(RelationStatistics).init(allocator),
            .entity_statistics = StringHashMap(EntityStatistics).init(allocator),
            .is_running = true,
        };

        try pipeline.initializeDefaultPatterns();
        return pipeline;
    }

    fn initializeDefaultPatterns(self: *CREVPipeline) !void {
        try self.relation_patterns.append(.{ .pattern = " is a ", .relation_type = "is_a", .weight = 0.9 });
        try self.relation_patterns.append(.{ .pattern = " is ", .relation_type = "is", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " has ", .relation_type = "has", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " contains ", .relation_type = "contains", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " belongs to ", .relation_type = "belongs_to", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " part of ", .relation_type = "part_of", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " located in ", .relation_type = "located_in", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " works at ", .relation_type = "works_at", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " created ", .relation_type = "created", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " owns ", .relation_type = "owns", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " uses ", .relation_type = "uses", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " produces ", .relation_type = "produces", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " causes ", .relation_type = "causes", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " leads to ", .relation_type = "leads_to", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " related to ", .relation_type = "related_to", .weight = 0.5 });
    }

    pub fn deinit(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.deinit();
        self.knowledge_index.deinit();
        self.relation_patterns.deinit();

        var rel_iter = self.relation_statistics.iterator();
        while (rel_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.relation_statistics.deinit();

        var ent_iter = self.entity_statistics.iterator();
        while (ent_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.entity_statistics.deinit();
    }

    pub fn processTextStream(self: *CREVPipeline, text: []const u8) !PipelineResult {
        const start_ns = std.time.nanoTimestamp();
        var result = PipelineResult.init();

        var triplets = try self.extractTriplets(text);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = validation_result.confidence_adjusted;

                if (validation_result.hasConflicts()) {
                    const resolved = try self.resolveConflicts(triplet, validation_result.conflicts.items);
                    if (resolved != triplet) {
                        triplet.deinit();
                        self.allocator.destroy(triplet);
                    }
                    result.conflicts_resolved += validation_result.conflictCount();
                    self.conflict_count += validation_result.conflictCount();
                }

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = std.time.nanoTimestamp();
        result.processing_time_ns = @intCast(end_ns - start_ns);
        result.stage = .indexing;
        return result;
    }

    pub fn processStructuredDataStream(self: *CREVPipeline, data: []const u8) !PipelineResult {
        const start_ns = std.time.nanoTimestamp();
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromStructured(data);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = validation_result.confidence_adjusted;

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = std.time.nanoTimestamp();
        result.processing_time_ns = @intCast(end_ns - start_ns);
        result.stage = .indexing;
        return result;
    }

    pub fn processImageMetadataStream(self: *CREVPipeline, metadata: []const u8) !PipelineResult {
        const start_ns = std.time.nanoTimestamp();
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromImageMetadata(metadata);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = std.time.nanoTimestamp();
        result.processing_time_ns = @intCast(end_ns - start_ns);
        result.stage = .indexing;
        return result;
    }

    pub fn extractTriplets(self: *CREVPipeline, text: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);

        var sentences = ArrayList([]const u8).init(self.allocator);
        defer sentences.deinit();

        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            if (c == '.' or c == '!' or c == '?' or c == '\n') {
                if (i > start) {
                    const sentence = std.mem.trim(u8, text[start..i], " \t\r\n");
                    if (sentence.len >= self.tokenizer_config.min_entity_length) {
                        try sentences.append(sentence);
                    }
                }
                start = i + 1;
            }
        }
        if (start < text.len) {
            const sentence = std.mem.trim(u8, text[start..], " \t\r\n");
            if (sentence.len >= self.tokenizer_config.min_entity_length) {
                try sentences.append(sentence);
            }
        }

        for (sentences.items) |sentence| {
            for (self.relation_patterns.items) |pattern| {
                if (std.mem.indexOf(u8, sentence, pattern.pattern)) |rel_pos| {
                    const subject = std.mem.trim(u8, sentence[0..rel_pos], " \t\r\n");
                    const object_start = rel_pos + pattern.pattern.len;
                    if (object_start < sentence.len) {
                        const object = std.mem.trim(u8, sentence[object_start..], " \t\r\n.,;:!?");

                        if (subject.len >= self.tokenizer_config.min_entity_length and
                            subject.len <= self.tokenizer_config.max_entity_length and
                            object.len >= self.tokenizer_config.min_entity_length and
                            object.len <= self.tokenizer_config.max_entity_length)
                        {
                            const confidence = pattern.weight * self.computeConfidence(subject, object);
                            if (confidence >= self.tokenizer_config.min_confidence_threshold) {
                                const triplet = try self.allocator.create(RelationalTriplet);
                                triplet.* = try RelationalTriplet.init(
                                    self.allocator,
                                    subject,
                                    pattern.relation_type,
                                    object,
                                    confidence,
                                );
                                try triplets.append(triplet);
                            }
                        }
                    }
                }
            }
        }

        return triplets;
    }

    fn extractTripletsFromStructured(self: *CREVPipeline, data: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);

        var lines = std.mem.split(u8, data, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            var parts = std.mem.split(u8, trimmed, ",");
            var fields = ArrayList([]const u8).init(self.allocator);
            defer fields.deinit();

            while (parts.next()) |part| {
                try fields.append(std.mem.trim(u8, part, " \t\""));
            }

            if (fields.items.len >= 3) {
                const triplet = try self.allocator.create(RelationalTriplet);
                triplet.* = try RelationalTriplet.init(
                    self.allocator,
                    fields.items[0],
                    fields.items[1],
                    fields.items[2],
                    if (fields.items.len >= 4) std.fmt.parseFloat(f64, fields.items[3]) catch 0.8 else 0.8,
                );
                try triplets.append(triplet);
            }
        }

        return triplets;
    }

    fn extractTripletsFromImageMetadata(self: *CREVPipeline, metadata: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);

        var lines = std.mem.split(u8, metadata, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
                const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t");
                const value = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");

                if (key.len > 0 and value.len > 0) {
                    const triplet = try self.allocator.create(RelationalTriplet);
                    triplet.* = try RelationalTriplet.init(
                        self.allocator,
                        "image",
                        key,
                        value,
                        0.9,
                    );
                    try triplet.setMetadata("source_type", "image_metadata");
                    try triplets.append(triplet);
                }
            }
        }

        return triplets;
    }

    fn computeConfidence(self: *CREVPipeline, subject: []const u8, object: []const u8) f64 {
        _ = self;
        var confidence: f64 = 1.0;

        const subject_len = @as(f64, @floatFromInt(subject.len));
        const object_len = @as(f64, @floatFromInt(object.len));

        if (subject_len < 3 or object_len < 3) {
            confidence *= 0.7;
        }

        if (subject_len > 50 or object_len > 50) {
            confidence *= 0.85;
        }

        var subject_upper: usize = 0;
        for (subject) |c| {
            if (c >= 'A' and c <= 'Z') subject_upper += 1;
        }
        if (subject_upper > 0 and subject_upper == subject.len) {
            confidence *= 0.9;
        } else if (subject.len > 0 and subject[0] >= 'A' and subject[0] <= 'Z') {
            confidence *= 1.1;
            if (confidence > 1.0) confidence = 1.0;
        }

        return confidence;
    }

    pub fn validateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !ValidationResult {
        var result = ValidationResult.init(self.allocator, triplet);

        if (triplet.subject.len < self.tokenizer_config.min_entity_length or
            triplet.object.len < self.tokenizer_config.min_entity_length)
        {
            result.is_valid = false;
            result.setValidationMethod("length_check");
            return result;
        }

        if (triplet.confidence < self.validation_threshold) {
            result.is_valid = false;
            result.confidence_adjusted = triplet.confidence;
            result.setValidationMethod("confidence_threshold");
            return result;
        }

        var existing_triplets = try self.knowledge_index.query(triplet.subject, null, triplet.object, self.allocator);
        defer existing_triplets.deinit();

        for (existing_triplets.items) |existing| {
            if (!self.checkConsistency(triplet, existing)) {
                try result.addConflict(existing);
            }
        }

        result.anomaly_score = try self.computeAnomalyScore(triplet);

        if (result.anomaly_score > 0.85) {
            result.is_valid = false;
            result.setValidationMethod("anomaly_detection");
            return result;
        }

        result.confidence_adjusted = triplet.confidence * (1.0 - result.anomaly_score * 0.3);
        if (result.hasConflicts()) {
            result.confidence_adjusted *= 0.9;
        }

        result.setValidationMethod("full_validation");
        return result;
    }

    fn computeAnomalyScore(self: *CREVPipeline, triplet: *RelationalTriplet) !f64 {
        var anomaly_score: f64 = 0.0;
        var factors: f64 = 0.0;

        if (self.relation_statistics.get(triplet.relation)) |stats| {
            if (stats.count > 10) {
                const std_dev = stats.getStdDev();
                if (std_dev > 0.01) {
                    const z_score = @fabs(triplet.confidence - stats.avg_confidence) / std_dev;
                    const stat_anomaly = @min(1.0, z_score / 3.0);
                    anomaly_score += stat_anomaly * 0.3;
                    factors += 0.3;
                }
            }
        }

        const subject_known = self.entity_statistics.contains(triplet.subject);
        const object_known = self.entity_statistics.contains(triplet.object);

        if (!subject_known and !object_known) {
            anomaly_score += 0.4;
            factors += 0.4;
        } else if (!subject_known or !object_known) {
            anomaly_score += 0.2;
            factors += 0.2;
        }

        if (!self.relation_statistics.contains(triplet.relation)) {
            anomaly_score += 0.15;
            factors += 0.15;
        }

        if (factors > 0) {
            return anomaly_score / factors;
        }
        return 0.0;
    }

    pub fn checkConsistency(self: *CREVPipeline, triplet: *RelationalTriplet, existing: *RelationalTriplet) bool {
        _ = self;

        if (std.mem.eql(u8, triplet.relation, existing.relation)) {
            return true;
        }

        const contradicting_pairs = [_][2][]const u8{
            .{ "is_a", "is_not" },
            .{ "has", "lacks" },
            .{ "owns", "does_not_own" },
            .{ "contains", "excludes" },
            .{ "causes", "prevents" },
        };

        for (contradicting_pairs) |pair| {
            if ((std.mem.eql(u8, triplet.relation, pair[0]) and std.mem.eql(u8, existing.relation, pair[1])) or
                (std.mem.eql(u8, triplet.relation, pair[1]) and std.mem.eql(u8, existing.relation, pair[0])))
            {
                return false;
            }
        }

        return true;
    }

    pub fn resolveConflicts(
        self: *CREVPipeline,
        triplet: *RelationalTriplet,
        conflicts: []*RelationalTriplet,
    ) !*RelationalTriplet {
        if (conflicts.len == 0) {
            return triplet;
        }

        var best_triplet = triplet;
        var best_confidence = triplet.confidence;

        for (conflicts) |conflict| {
            if (conflict.confidence > best_confidence) {
                best_confidence = conflict.confidence;
                best_triplet = conflict;
            }
        }

        if (best_triplet != triplet) {
            const new_triplet = try self.allocator.create(RelationalTriplet);
            new_triplet.* = try best_triplet.clone(self.allocator);

            const total_confidence = triplet.confidence + best_triplet.confidence;
            new_triplet.confidence = best_triplet.confidence / total_confidence * best_triplet.confidence +
                triplet.confidence / total_confidence * triplet.confidence;

            return new_triplet;
        }

        return triplet;
    }

    pub fn integrateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        try self.knowledge_index.index(triplet);
        self.total_confidence_sum += triplet.confidence;

        try self.updateStatistics(triplet);

        const data = try std.fmt.allocPrint(self.allocator, "{s}|{s}|{s}|{d:.6}", .{
            triplet.subject,
            triplet.relation,
            triplet.object,
            triplet.confidence,
        });
        defer self.allocator.free(data);

        _ = try self.kernel.allocateMemory(data, null);
    }

    fn updateStatistics(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        var rel_result = try self.relation_statistics.getOrPut(triplet.relation);
        if (!rel_result.found_existing) {
            rel_result.key_ptr.* = try self.allocator.dupe(u8, triplet.relation);
            rel_result.value_ptr.* = RelationStatistics.init();
        }
        rel_result.value_ptr.update(triplet.confidence);

        var subj_result = try self.entity_statistics.getOrPut(triplet.subject);
        if (!subj_result.found_existing) {
            subj_result.key_ptr.* = try self.allocator.dupe(u8, triplet.subject);
            subj_result.value_ptr.* = EntityStatistics.init();
        }
        subj_result.value_ptr.count += 1;
        subj_result.value_ptr.as_subject += 1;
        subj_result.value_ptr.total_confidence += triplet.confidence;

        var obj_result = try self.entity_statistics.getOrPut(triplet.object);
        if (!obj_result.found_existing) {
            obj_result.key_ptr.* = try self.allocator.dupe(u8, triplet.object);
            obj_result.value_ptr.* = EntityStatistics.init();
        }
        obj_result.value_ptr.count += 1;
        obj_result.value_ptr.as_object += 1;
        obj_result.value_ptr.total_confidence += triplet.confidence;
    }

    pub fn queryKnowledgeGraph(
        self: *CREVPipeline,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
    ) !ArrayList(*RelationalTriplet) {
        return self.knowledge_index.query(subject, relation, object, self.allocator);
    }

    pub fn getPipelineStatistics(self: *CREVPipeline) PipelineStatistics {
        const uptime = std.time.milliTimestamp() - self.start_time;
        const uptime_sec = @as(f64, @floatFromInt(@max(1, uptime))) / 1000.0;

        return PipelineStatistics{
            .total_extractions = self.extraction_count,
            .total_validations = self.validation_count,
            .total_integrations = self.integration_count,
            .average_confidence = if (self.integration_count > 0)
                self.total_confidence_sum / @as(f64, @floatFromInt(self.integration_count))
            else
                0.0,
            .conflict_rate = if (self.validation_count > 0)
                @as(f64, @floatFromInt(self.conflict_count)) / @as(f64, @floatFromInt(self.validation_count))
            else
                0.0,
            .throughput = @as(f64, @floatFromInt(self.integration_count)) / uptime_sec,
            .buffer_utilization = self.triplet_buffer.getUtilization(),
            .unique_subjects = self.knowledge_index.getUniqueSubjects(),
            .unique_relations = self.knowledge_index.getUniqueRelations(),
            .unique_objects = self.knowledge_index.getUniqueObjects(),
            .uptime_ms = uptime,
        };
    }

    pub fn shutdown(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.clear();
    }

    pub fn addRelationPattern(self: *CREVPipeline, pattern: []const u8, relation_type: []const u8, weight: f64) !void {
        try self.relation_patterns.append(.{
            .pattern = pattern,
            .relation_type = relation_type,
            .weight = weight,
        });
    }

    pub fn setValidationThreshold(self: *CREVPipeline, threshold: f64) void {
        self.validation_threshold = @max(0.0, @min(1.0, threshold));
    }

    pub fn getKnowledgeGraphSize(self: *CREVPipeline) usize {
        return self.knowledge_index.count();
    }

    pub fn isRunning(self: *const CREVPipeline) bool {
        return self.is_running;
    }
};

test "ExtractionStage toString and fromString" {
    const testing = std.testing;

    try testing.expectEqualStrings("tokenization", ExtractionStage.tokenization.toString());
    try testing.expectEqualStrings("triplet_extraction", ExtractionStage.triplet_extraction.toString());
    try testing.expectEqualStrings("validation", ExtractionStage.validation.toString());
    try testing.expectEqualStrings("integration", ExtractionStage.integration.toString());
    try testing.expectEqualStrings("indexing", ExtractionStage.indexing.toString());

    try testing.expectEqual(ExtractionStage.tokenization, ExtractionStage.fromString("tokenization").?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.fromString("validation").?);
    try testing.expect(ExtractionStage.fromString("invalid") == null);
}

test "ExtractionStage next" {
    const testing = std.testing;

    try testing.expectEqual(ExtractionStage.triplet_extraction, ExtractionStage.tokenization.next().?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.triplet_extraction.next().?);
    try testing.expectEqual(ExtractionStage.integration, ExtractionStage.validation.next().?);
    try testing.expectEqual(ExtractionStage.indexing, ExtractionStage.integration.next().?);
    try testing.expect(ExtractionStage.indexing.next() == null);
}

test "RelationalTriplet initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    defer triplet.deinit();

    try testing.expectEqualStrings("Alice", triplet.subject);
    try testing.expectEqualStrings("knows", triplet.relation);
    try testing.expectEqualStrings("Bob", triplet.object);
    try testing.expectApproxEqAbs(@as(f64, 0.9), triplet.confidence, 0.001);
}

test "RelationalTriplet clone" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var original = try RelationalTriplet.init(allocator, "Paris", "is_a", "City", 0.95);
    defer original.deinit();

    try original.setMetadata("source", "test");

    var cloned = try original.clone(allocator);
    defer cloned.deinit();

    try testing.expectEqualStrings(original.subject, cloned.subject);
    try testing.expectEqualStrings(original.relation, cloned.relation);
    try testing.expectEqualStrings(original.object, cloned.object);
    try testing.expectApproxEqAbs(original.confidence, cloned.confidence, 0.001);
    try testing.expectEqualStrings("test", cloned.getMetadata("source").?);
}

test "RelationalTriplet computeHash" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet2.deinit();

    const hash1 = triplet1.computeHash();
    const hash2 = triplet2.computeHash();

    try testing.expect(hash1.len == 32);
    try testing.expect(hash2.len == 32);
}

test "RelationalTriplet equals" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.8);
    defer triplet2.deinit();

    var triplet3 = try RelationalTriplet.init(allocator, "A", "different", "B", 0.9);
    defer triplet3.deinit();

    try testing.expect(triplet1.equals(&triplet2));
    try testing.expect(!triplet1.equals(&triplet3));
}

test "KnowledgeGraphIndex initialization and indexing" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Entity1", "related_to", "Entity2", 0.8);

    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());
}

test "KnowledgeGraphIndex query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    try index.index(triplet1);

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "Alice", "works_at", "Company", 0.85);
    try index.index(triplet2);

    var results = try index.query("Alice", null, null, allocator);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "KnowledgeGraphIndex queryBySubject" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubject", "has", "TestObject", 0.7);
    try index.index(triplet);

    const results = index.queryBySubject("TestSubject");
    try testing.expectEqual(@as(usize, 1), results.len);

    const empty_results = index.queryBySubject("NonExistent");
    try testing.expectEqual(@as(usize, 0), empty_results.len);
}

test "KnowledgeGraphIndex remove" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "ToRemove", "relation", "Target", 0.6);
    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());

    const removed = index.remove(triplet);
    try testing.expect(removed);
    try testing.expectEqual(@as(usize, 0), index.count());

    triplet.deinit();
    allocator.destroy(triplet);
}

test "StreamBuffer push and pop" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.isEmpty());
    try testing.expect(!buffer.isFull());

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    _ = try buffer.push(triplet1);

    try testing.expect(!buffer.isEmpty());
    try testing.expectEqual(@as(usize, 1), buffer.getSize());

    const popped = buffer.pop();
    try testing.expect(popped != null);
    try testing.expectEqualStrings("A", popped.?.subject);
    try testing.expect(buffer.isEmpty());

    popped.?.deinit();
    allocator.destroy(popped.?);
}

test "StreamBuffer capacity" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 3);
    defer buffer.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    _ = try buffer.push(triplet1);

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    _ = try buffer.push(triplet2);

    const triplet3 = try allocator.create(RelationalTriplet);
    triplet3.* = try RelationalTriplet.init(allocator, "3", "r", "c", 0.5);
    _ = try buffer.push(triplet3);

    try testing.expect(buffer.isFull());

    const triplet4 = try allocator.create(RelationalTriplet);
    triplet4.* = try RelationalTriplet.init(allocator, "4", "r", "d", 0.5);
    const success = try buffer.push(triplet4);
    try testing.expect(!success);
    try testing.expectEqual(@as(usize, 1), buffer.overflow_count);

    triplet4.deinit();
    allocator.destroy(triplet4);
}

test "StreamBuffer peek" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.peek() == null);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Peek", "test", "value", 0.7);
    _ = try buffer.push(triplet);

    const peeked = buffer.peek();
    try testing.expect(peeked != null);
    try testing.expectEqualStrings("Peek", peeked.?.subject);
    try testing.expectEqual(@as(usize, 1), buffer.getSize());
}

test "PipelineResult merge" {
    const testing = std.testing;

    var result1 = PipelineResult.init();
    result1.triplets_extracted = 10;
    result1.triplets_validated = 8;
    result1.triplets_integrated = 7;

    var result2 = PipelineResult.init();
    result2.triplets_extracted = 5;
    result2.triplets_validated = 4;
    result2.triplets_integrated = 3;

    result1.merge(result2);

    try testing.expectEqual(@as(usize, 15), result1.triplets_extracted);
    try testing.expectEqual(@as(usize, 12), result1.triplets_validated);
    try testing.expectEqual(@as(usize, 10), result1.triplets_integrated);
}

test "CREVPipeline initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.is_running);
    try testing.expectEqual(@as(usize, 0), pipeline.extraction_count);
    try testing.expectApproxEqAbs(@as(f64, 0.5), pipeline.validation_threshold, 0.001);
}

test "CREVPipeline extractTriplets" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Paris is a city. The Eiffel Tower is located in Paris.";
    var triplets = try pipeline.extractTriplets(text);
    defer {
        for (triplets.items) |triplet| {
            triplet.deinit();
            allocator.destroy(triplet);
        }
        triplets.deinit();
    }

    try testing.expect(triplets.items.len > 0);
}

test "CREVPipeline processTextStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Python is a programming language. Python has modules.";
    const result = try pipeline.processTextStream(text);

    try testing.expect(result.triplets_extracted > 0);
    try testing.expect(result.processing_time_ns > 0);
}

test "CREVPipeline processStructuredDataStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const data = "Alice,knows,Bob,0.9\nBob,works_at,Company,0.85";
    const result = try pipeline.processStructuredDataStream(data);

    try testing.expect(result.triplets_extracted == 2);
}

test "CREVPipeline validateTriplet" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    pipeline.setValidationThreshold(0.4);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubjectEntity", "is_a", "TestObjectEntity", 0.95);
    defer {
        triplet.deinit();
        allocator.destroy(triplet);
    }

    var result = try pipeline.validateTriplet(triplet);
    defer result.deinit();

    try testing.expect(result.confidence_adjusted > 0);
}

test "CREVPipeline checkConsistency" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    var triplet1 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.8);
    defer triplet2.deinit();

    try testing.expect(pipeline.checkConsistency(&triplet1, &triplet2));

    var triplet3 = try RelationalTriplet.init(allocator, "A", "is_not", "B", 0.7);
    defer triplet3.deinit();

    try testing.expect(!pipeline.checkConsistency(&triplet1, &triplet3));
}

test "CREVPipeline getPipelineStatistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const stats = pipeline.getPipelineStatistics();

    try testing.expectEqual(@as(usize, 0), stats.total_extractions);
    try testing.expectEqual(@as(usize, 0), stats.total_validations);
    try testing.expect(stats.uptime_ms >= 0);
}

test "CREVPipeline queryKnowledgeGraph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "DirectEntity", "has_property", "TestProperty", 0.9);
    try pipeline.knowledge_index.index(triplet);

    var results = try pipeline.queryKnowledgeGraph("DirectEntity", null, null);
    defer results.deinit();

    try testing.expect(results.items.len > 0);
}

test "CREVPipeline shutdown" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.isRunning());
    pipeline.shutdown();
    try testing.expect(!pipeline.isRunning());
}

test "ValidationResult initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "S", "R", "O", 0.75);
    defer triplet.deinit();

    var result = ValidationResult.init(allocator, &triplet);
    defer result.deinit();

    try testing.expect(result.is_valid);
    try testing.expect(!result.hasConflicts());
    try testing.expectEqual(@as(usize, 0), result.conflictCount());
}

test "RelationStatistics update" {
    const testing = std.testing;

    var stats = CREVPipeline.RelationStatistics.init();

    stats.update(0.8);
    try testing.expectEqual(@as(usize, 1), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.8), stats.avg_confidence, 0.001);

    stats.update(0.6);
    try testing.expectEqual(@as(usize, 2), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.7), stats.avg_confidence, 0.001);

    try testing.expect(stats.getVariance() >= 0);
    try testing.expect(stats.getStdDev() >= 0);
}

test "StreamBuffer utilization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 10);
    defer buffer.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.0), buffer.getUtilization(), 0.001);

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    _ = try buffer.push(triplet1);

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    _ = try buffer.push(triplet2);

    try testing.expectApproxEqAbs(@as(f64, 0.2), buffer.getUtilization(), 0.001);
}
