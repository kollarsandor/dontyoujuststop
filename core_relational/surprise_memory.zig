
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;

const chaos = @import("chaos_core.zig");
const MemoryBlock = chaos.MemoryBlock;
const MemoryBlockState = chaos.MemoryBlockState;
const ContentAddressableStorage = chaos.ContentAddressableStorage;
const DataFlowAnalyzer = chaos.DataFlowAnalyzer;

pub const SurpriseMetrics = struct {
    jaccard_dissimilarity: f64,
    content_hash_distance: f64,
    temporal_novelty: f64,
    combined_surprise: f64,

    pub fn init(jaccard: f64, hash_dist: f64, temporal: f64) SurpriseMetrics {
        const combined = (jaccard + hash_dist + temporal) / 3.0;
        return SurpriseMetrics{
            .jaccard_dissimilarity = jaccard,
            .content_hash_distance = hash_dist,
            .temporal_novelty = temporal,
            .combined_surprise = combined,
        };
    }

    pub fn exceedsThreshold(self: *const SurpriseMetrics, threshold: f64) bool {
        return self.combined_surprise > threshold;
    }
};

pub const SurpriseRecord = struct {
    block_id: [16]u8,
    surprise_score: f64,
    creation_time: i64,
    last_update_time: i64,
    retention_priority: f64,
    access_frequency: usize,

    pub fn init(block_id: [16]u8, score: f64) SurpriseRecord {
        const now = std.time.milliTimestamp();
        return SurpriseRecord{
            .block_id = block_id,
            .surprise_score = score,
            .creation_time = now,
            .last_update_time = now,
            .retention_priority = score,
            .access_frequency = 1,
        };
    }

    pub fn updateRetention(self: *SurpriseRecord) void {
        const age = @as(f64, @floatFromInt(std.time.milliTimestamp() - self.creation_time));
        const age_factor = 1.0 / (1.0 + age / 1000000.0);
        const frequency_factor = @log(@as(f64, @floatFromInt(self.access_frequency + 1)));
        self.retention_priority = self.surprise_score * (0.5 + 0.3 * age_factor + 0.2 * frequency_factor);
        self.last_update_time = std.time.milliTimestamp();
    }

    pub fn recordAccess(self: *SurpriseRecord) void {
        self.access_frequency += 1;
        self.updateRetention();
    }
};

pub const SurpriseMemoryStatistics = struct {
    total_blocks: usize,
    high_surprise_blocks: usize,
    low_surprise_blocks: usize,
    average_surprise: f64,
    surprise_threshold: f64,
    evictions_due_to_low_surprise: usize,
    novel_block_allocations: usize,

    pub fn init(threshold: f64) SurpriseMemoryStatistics {
        return SurpriseMemoryStatistics{
            .total_blocks = 0,
            .high_surprise_blocks = 0,
            .low_surprise_blocks = 0,
            .average_surprise = 0.0,
            .surprise_threshold = threshold,
            .evictions_due_to_low_surprise = 0,
            .novel_block_allocations = 0,
        };
    }
};

pub const SurpriseMemoryManager = struct {
    storage: *ContentAddressableStorage,
    flow_analyzer: *DataFlowAnalyzer,
    surprise_records: std.HashMap([16]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage),
    surprise_threshold: f64,
    statistics: SurpriseMemoryStatistics,
    allocator: Allocator,

    const Self = @This();
    const DEFAULT_SURPRISE_THRESHOLD: f64 = 0.3;

    pub fn init(allocator: Allocator, storage: *ContentAddressableStorage, analyzer: *DataFlowAnalyzer) Self {
        return Self{
            .storage = storage,
            .flow_analyzer = analyzer,
            .surprise_records = std.HashMap([16]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .surprise_threshold = DEFAULT_SURPRISE_THRESHOLD,
            .statistics = SurpriseMemoryStatistics.init(DEFAULT_SURPRISE_THRESHOLD),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.surprise_records.deinit();
    }

    pub fn setSurpriseThreshold(self: *Self, threshold: f64) void {
        self.surprise_threshold = threshold;
        self.statistics.surprise_threshold = threshold;
    }

    pub fn computeSurprise(self: *Self, new_data: []const u8) !SurpriseMetrics {
        const new_hash = self.computeContentHash(new_data);

        var min_jaccard_dist: f64 = 1.0;
        var min_hash_dist: f64 = 1.0;

        var storage_iter = self.storage.storage.iterator();
        while (storage_iter.next()) |entry| {
            const existing_block = entry.value_ptr;
            const jaccard = self.computeJaccardDistance(new_data, existing_block.data);
            if (jaccard < min_jaccard_dist) {
                min_jaccard_dist = jaccard;
            }

            const hash_dist = self.computeHashDistance(new_hash, existing_block.content_hash);
            if (hash_dist < min_hash_dist) {
                min_hash_dist = hash_dist;
            }
        }

        const temporal_novelty = if (self.storage.storage.count() > 0) 1.0 / @sqrt(@as(f64, @floatFromInt(self.storage.storage.count()))) else 1.0;

        return SurpriseMetrics.init(min_jaccard_dist, min_hash_dist, temporal_novelty);
    }

    fn computeContentHash(self: *Self, data: []const u8) [16]u8 {
        _ = self;
        var hash_out: [32]u8 = undefined;
        Sha256.hash(data, &hash_out, .{});
        var result: [16]u8 = undefined;
        @memcpy(&result, hash_out[0..16]);
        return result;
    }

    fn computeJaccardDistance(self: *Self, data_a: []const u8, data_b: []const u8) f64 {
        var set_a = std.AutoHashMap(u8, void).init(self.allocator);
        defer set_a.deinit();
        var set_b = std.AutoHashMap(u8, void).init(self.allocator);
        defer set_b.deinit();

        for (data_a) |byte| {
            set_a.put(byte, {}) catch continue;
        }
        for (data_b) |byte| {
            set_b.put(byte, {}) catch continue;
        }

        var intersection_count: usize = 0;
        var iter = set_a.iterator();
        while (iter.next()) |entry| {
            if (set_b.contains(entry.key_ptr.*)) {
                intersection_count += 1;
            }
        }

        const union_count = set_a.count() + set_b.count() - intersection_count;
        if (union_count == 0) return 0.0;

        const jaccard_similarity = @as(f64, @floatFromInt(intersection_count)) / @as(f64, @floatFromInt(union_count));
        return 1.0 - jaccard_similarity;
    }

    fn computeHashDistance(self: *Self, hash_a: [16]u8, hash_b: [16]u8) f64 {
        _ = self;
        var hamming_dist: usize = 0;
        var byte_idx: usize = 0;
        while (byte_idx < hash_a.len) : (byte_idx += 1) {
            hamming_dist += @popCount(hash_a[byte_idx] ^ hash_b[byte_idx]);
        }
        const max_hamming = 16 * 8;
        return @as(f64, @floatFromInt(hamming_dist)) / @as(f64, @floatFromInt(max_hamming));
    }

    pub fn storeWithSurprise(self: *Self, data: []const u8, preferred_core: ?usize) ![16]u8 {
        const surprise = try self.computeSurprise(data);

        if (surprise.exceedsThreshold(self.surprise_threshold)) {
            const block_id = try self.storage.store(data, preferred_core);

            const record = SurpriseRecord.init(block_id, surprise.combined_surprise);
            try self.surprise_records.put(block_id, record);

            self.statistics.novel_block_allocations += 1;
            self.statistics.high_surprise_blocks += 1;
            self.updateStatistics();

            return block_id;
        } else {
            const existing_block_id = self.storage.retrieveByContent(data);
            if (existing_block_id) |block_id| {
                if (self.surprise_records.getPtr(block_id)) |record| {
                    record.recordAccess();
                }
                return block_id;
            }

            const block_id = try self.storage.store(data, preferred_core);
            const record = SurpriseRecord.init(block_id, surprise.combined_surprise);
            try self.surprise_records.put(block_id, record);

            self.statistics.low_surprise_blocks += 1;
            self.updateStatistics();

            return block_id;
        }
    }

    pub fn evictLowSurpriseBlocks(self: *Self, target_capacity: usize) !usize {
        var candidates = ArrayList(struct { block_id: [16]u8, priority: f64 }).init(self.allocator);
        defer candidates.deinit();

        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.updateRetention();
            try candidates.append(.{
                .block_id = entry.key_ptr.*,
                .priority = entry.value_ptr.retention_priority,
            });
        }

        const CandidateItem = struct { block_id: [16]u8, priority: f64 };
        std.mem.sort(CandidateItem, candidates.items, {}, struct {
            fn lessThan(_: void, a: CandidateItem, b: CandidateItem) bool {
                return a.priority < b.priority;
            }
        }.lessThan);

        var evicted_count: usize = 0;
        const current_size = self.storage.storage.count();
        if (current_size <= target_capacity) return 0;

        const to_evict = current_size - target_capacity;
        for (candidates.items[0..@min(to_evict, candidates.items.len)]) |candidate| {
            if (self.storage.containsBlock(candidate.block_id)) {
                self.storage.removeBlock(candidate.block_id);
                _ = self.surprise_records.remove(candidate.block_id);
                evicted_count += 1;
                self.statistics.evictions_due_to_low_surprise += 1;
            }
        }

        self.updateStatistics();
        return evicted_count;
    }

    fn updateStatistics(self: *Self) void {
        self.statistics.total_blocks = self.surprise_records.count();

        var total_surprise: f64 = 0.0;
        var high_count: usize = 0;
        var low_count: usize = 0;

        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            total_surprise += entry.value_ptr.surprise_score;
            if (entry.value_ptr.surprise_score > self.surprise_threshold) {
                high_count += 1;
            } else {
                low_count += 1;
            }
        }

        if (self.surprise_records.count() > 0) {
            self.statistics.average_surprise = total_surprise / @as(f64, @floatFromInt(self.surprise_records.count()));
        }
        self.statistics.high_surprise_blocks = high_count;
        self.statistics.low_surprise_blocks = low_count;
    }

    pub fn organizeByEntanglement(self: *Self) !void {
        var high_surprise_blocks = ArrayList([16]u8).init(self.allocator);
        defer high_surprise_blocks.deinit();

        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.surprise_score > self.surprise_threshold) {
                try high_surprise_blocks.append(entry.key_ptr.*);
            }
        }

        var i: usize = 0;
        while (i < high_surprise_blocks.items.len) : (i += 1) {
            var j = i + 1;
            while (j < high_surprise_blocks.items.len) : (j += 1) {
                const block1 = high_surprise_blocks.items[i];
                const block2 = high_surprise_blocks.items[j];
                _ = try self.storage.entangleBlocks(block1, block2);
            }
        }
    }

    pub fn getStatistics(self: *const Self) SurpriseMemoryStatistics {
        return self.statistics;
    }

    pub fn getSurpriseRecord(self: *const Self, block_id: [16]u8) ?SurpriseRecord {
        return self.surprise_records.get(block_id);
    }
};

test "surprise_memory_basic" {
    const allocator = std.testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var manager = SurpriseMemoryManager.init(allocator, &storage, &analyzer);
    defer manager.deinit();

    const data1 = "unique_data_content_1";
    const data2 = "unique_data_content_2";

    const block1 = try manager.storeWithSurprise(data1, null);
    const block2 = try manager.storeWithSurprise(data2, null);

    try std.testing.expect(!std.mem.eql(u8, &block1, &block2));

    const stats = manager.getStatistics();
    try std.testing.expect(stats.total_blocks >= 2);
}
