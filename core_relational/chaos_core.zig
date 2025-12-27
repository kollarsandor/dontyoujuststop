const std = @import("std");
const nsir = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const ChaosCoreConfig = struct {
    pub const BLOCK_ID_SIZE: usize = 16;
    pub const CONTENT_HASH_SIZE: usize = 16;
    pub const SHA256_DIGEST_SIZE: usize = 32;
    pub const ENTROPY_BITS_SHIFT: u5 = 24;
    pub const DEFAULT_BLOCK_SIZE: usize = 1048576;
};

pub const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
pub const Node = nsir.Node;
pub const Edge = nsir.Edge;
pub const EdgeQuality = nsir.EdgeQuality;

pub const MemoryBlockState = enum(u8) {
    free = 0,
    allocated = 1,
    entangled = 2,
    migrating = 3,

    pub fn toString(self: MemoryBlockState) []const u8 {
        return switch (self) {
            .free => "free",
            .allocated => "allocated",
            .entangled => "entangled",
            .migrating => "migrating",
        };
    }

    pub fn fromString(s: []const u8) ?MemoryBlockState {
        if (std.mem.eql(u8, s, "free")) return .free;
        if (std.mem.eql(u8, s, "allocated")) return .allocated;
        if (std.mem.eql(u8, s, "entangled")) return .entangled;
        if (std.mem.eql(u8, s, "migrating")) return .migrating;
        return null;
    }
};

pub const BlockIdContext = struct {
    pub fn hash(_: @This(), key: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, b: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return std.mem.eql(u8, &a, &b);
    }
};

pub const BlockIdSet = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, void, BlockIdContext, std.hash_map.default_max_load_percentage);

pub const MemoryBlock = struct {
    block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    content_hash: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    data: []const u8,
    size: usize,
    state: MemoryBlockState,
    affinity_core: ?usize,
    access_count: usize,
    last_access_time: i64,
    entangled_blocks: BlockIdSet,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        content_hash: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        data: []const u8,
        preferred_core: ?usize,
    ) !MemoryBlock {
        return MemoryBlock{
            .block_id = block_id,
            .content_hash = content_hash,
            .data = try allocator.dupe(u8, data),
            .size = data.len,
            .state = .allocated,
            .affinity_core = preferred_core,
            .access_count = 1,
            .last_access_time = std.time.milliTimestamp(),
            .entangled_blocks = BlockIdSet.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryBlock) void {
        self.allocator.free(self.data);
        self.entangled_blocks.deinit();
    }

    pub fn updateAccessTime(self: *MemoryBlock) void {
        self.access_count += 1;
        self.last_access_time = std.time.milliTimestamp();
    }

    pub fn addEntangledBlock(self: *MemoryBlock, other_block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !void {
        try self.entangled_blocks.put(other_block_id, {});
    }

    pub fn hasEntanglement(self: *const MemoryBlock, other_block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return self.entangled_blocks.contains(other_block_id);
    }

    pub fn entangledCount(self: *const MemoryBlock) usize {
        return self.entangled_blocks.count();
    }

    pub fn clone(self: *const MemoryBlock, allocator: Allocator) !MemoryBlock {
        var new_block = MemoryBlock{
            .block_id = self.block_id,
            .content_hash = self.content_hash,
            .data = try allocator.dupe(u8, self.data),
            .size = self.size,
            .state = self.state,
            .affinity_core = self.affinity_core,
            .access_count = self.access_count,
            .last_access_time = self.last_access_time,
            .entangled_blocks = BlockIdSet.init(allocator),
            .allocator = allocator,
        };
        var iter = self.entangled_blocks.iterator();
        while (iter.next()) |entry| {
            try new_block.entangled_blocks.put(entry.key_ptr.*, {});
        }
        return new_block;
    }
};

pub const TaskDescriptor = struct {
    task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8,
    priority: i32,
    data_dependencies: ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8),
    estimated_cycles: usize,
    assigned_core: ?usize,
    completion_status: bool,
    start_time: i64,
    end_time: i64,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8,
        priority: i32,
        estimated_cycles: usize,
    ) TaskDescriptor {
        return TaskDescriptor{
            .task_id = task_id,
            .priority = priority,
            .data_dependencies = ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8).init(allocator),
            .estimated_cycles = estimated_cycles,
            .assigned_core = null,
            .completion_status = false,
            .start_time = 0,
            .end_time = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TaskDescriptor) void {
        self.data_dependencies.deinit();
    }

    pub fn addDependency(self: *TaskDescriptor, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !void {
        try self.data_dependencies.append(block_id);
    }

    pub fn getDuration(self: *const TaskDescriptor) i64 {
        if (self.completion_status and self.end_time > self.start_time) {
            return self.end_time - self.start_time;
        }
        return 0;
    }

    pub fn clone(self: *const TaskDescriptor, allocator: Allocator) !TaskDescriptor {
        var new_task = TaskDescriptor{
            .task_id = self.task_id,
            .priority = self.priority,
            .data_dependencies = ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8).init(allocator),
            .estimated_cycles = self.estimated_cycles,
            .assigned_core = self.assigned_core,
            .completion_status = self.completion_status,
            .start_time = self.start_time,
            .end_time = self.end_time,
            .allocator = allocator,
        };
        for (self.data_dependencies.items) |dep| {
            try new_task.data_dependencies.append(dep);
        }
        return new_task;
    }
};

pub const StorageStatistics = struct {
    total_blocks: usize,
    entangled_blocks: usize,
    used_capacity: usize,
    total_capacity: usize,
    utilization: f64,
    unique_contents: usize,
    cores_with_affinity: usize,
};

pub const SchedulerStatistics = struct {
    pending_tasks: usize,
    active_tasks: usize,
    completed_tasks: usize,
    average_completion_time: f64,
};

pub const KernelStatistics = struct {
    cycle_count: usize,
    migration_count: usize,
    storage: StorageStatistics,
    scheduler: SchedulerStatistics,
};

pub const CoreState = enum(u8) {
    idle = 0,
    active = 1,
    power_gated = 2,
    stalled = 3,
};

pub const ProcessingCore = struct {
    core_id: usize,
    state: CoreState,
    cycles_active: usize,
    cycles_idle: usize,

    pub fn init(core_id: usize) ProcessingCore {
        return ProcessingCore{
            .core_id = core_id,
            .state = .idle,
            .cycles_active = 0,
            .cycles_idle = 0,
        };
    }

    pub fn getWorkload(self: *const ProcessingCore) f64 {
        const total = self.cycles_active + self.cycles_idle;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cycles_active)) / @as(f64, @floatFromInt(total));
    }
};

pub const ContentAddressableStorage = struct {
    storage: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, MemoryBlock, BlockIdContext, std.hash_map.default_max_load_percentage),
    content_index: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage),
    affinity_map: AutoHashMap(usize, BlockIdSet),
    total_capacity: usize,
    used_capacity: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ContentAddressableStorage {
        return ContentAddressableStorage{
            .storage = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, MemoryBlock, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .content_index = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .affinity_map = AutoHashMap(usize, BlockIdSet).init(allocator),
            .total_capacity = ChaosCoreConfig.DEFAULT_BLOCK_SIZE,
            .used_capacity = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ContentAddressableStorage) void {
        var storage_iter = self.storage.iterator();
        while (storage_iter.next()) |entry| {
            var block = entry.value_ptr;
            block.deinit();
        }
        self.storage.deinit();

        var content_iter = self.content_index.iterator();
        while (content_iter.next()) |entry| {
            var block_set = entry.value_ptr;
            block_set.deinit();
        }
        self.content_index.deinit();

        var affinity_iter = self.affinity_map.iterator();
        while (affinity_iter.next()) |entry| {
            var block_set = entry.value_ptr;
            block_set.deinit();
        }
        self.affinity_map.deinit();
    }

    fn computeContentHash(data: []const u8) [ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &hash_out, .{});
        var result: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
        @memcpy(&result, hash_out[0..16]);
        return result;
    }

    fn computeBlockId(content_hash: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, timestamp: i64) [ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        var buffer: [24]u8 = undefined;
        @memcpy(buffer[0..16], &content_hash);
        std.mem.writeInt(i64, buffer[16..24], timestamp, .Little);
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(&buffer, &hash_out, .{});
        var result: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
        @memcpy(&result, hash_out[0..16]);
        return result;
    }

    pub fn store(self: *ContentAddressableStorage, data: []const u8, preferred_core: ?usize) ![ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        const content_hash = computeContentHash(data);

        if (self.content_index.getPtr(content_hash)) |existing_blocks| {
            var iter = existing_blocks.iterator();
            if (iter.next()) |first_entry| {
                const block_id = first_entry.key_ptr.*;
                if (self.storage.getPtr(block_id)) |block| {
                    block.updateAccessTime();
                    return block_id;
                }
            }
        }

        const timestamp = std.time.milliTimestamp();
        const block_id = computeBlockId(content_hash, timestamp);
        const data_size = data.len;

        if (self.used_capacity + data_size > self.total_capacity) {
            try self.evictLeastUsed(data_size);
        }

        var block = try MemoryBlock.init(self.allocator, block_id, content_hash, data, preferred_core);
        errdefer block.deinit();

        try self.storage.put(block_id, block);

        var result = try self.content_index.getOrPut(content_hash);
        if (!result.found_existing) {
            result.value_ptr.* = BlockIdSet.init(self.allocator);
        }
        try result.value_ptr.put(block_id, {});

        if (preferred_core) |core_id| {
            var affinity_result = try self.affinity_map.getOrPut(core_id);
            if (!affinity_result.found_existing) {
                affinity_result.value_ptr.* = BlockIdSet.init(self.allocator);
            }
            try affinity_result.value_ptr.put(block_id, {});
        }

        self.used_capacity += data_size;
        return block_id;
    }

    pub fn retrieve(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
        if (self.storage.getPtr(block_id)) |block| {
            block.updateAccessTime();
            return block.data;
        }
        return null;
    }

    pub fn retrieveByContent(self: *ContentAddressableStorage, data: []const u8) ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        const content_hash = computeContentHash(data);
        if (self.content_index.getPtr(content_hash)) |block_set| {
            var iter = block_set.iterator();
            if (iter.next()) |entry| {
                return entry.key_ptr.*;
            }
        }
        return null;
    }

    pub fn entangleBlocks(self: *ContentAddressableStorage, block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !bool {
        const block1_ptr = self.storage.getPtr(block_id1);
        const block2_ptr = self.storage.getPtr(block_id2);

        if (block1_ptr == null or block2_ptr == null) {
            return false;
        }

        try block1_ptr.?.addEntangledBlock(block_id2);
        try block2_ptr.?.addEntangledBlock(block_id1);
        block1_ptr.?.state = .entangled;
        block2_ptr.?.state = .entangled;
        return true;
    }

    pub fn findNearestCore(
        self: *ContentAddressableStorage,
        block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        cores: *AutoHashMap(usize, ProcessingCore),
    ) ?usize {
        const block_ptr = self.storage.getPtr(block_id);
        if (block_ptr == null) {
            return null;
        }
        const block = block_ptr.?;

        if (block.affinity_core) |core| {
            if (cores.get(core)) |core_info| {
                if (core_info.state != .power_gated) {
                    return core;
                }
            }
        }

        var min_distance: f64 = std.math.inf(f64);
        var nearest_core: ?usize = null;

        var core_iter = cores.iterator();
        while (core_iter.next()) |core_entry| {
            const core_id = core_entry.key_ptr.*;
            const core_info = core_entry.value_ptr.*;

            if (core_info.state == .power_gated) {
                continue;
            }

            var entangled_count: usize = 0;
            if (self.affinity_map.getPtr(core_id)) |core_blocks| {
                var entangle_iter = block.entangled_blocks.iterator();
                while (entangle_iter.next()) |ent_entry| {
                    if (core_blocks.contains(ent_entry.key_ptr.*)) {
                        entangled_count += 1;
                    }
                }
            }

            const distance = 1.0 / (1.0 + @as(f64, @floatFromInt(entangled_count)));
            if (distance < min_distance) {
                min_distance = distance;
                nearest_core = core_id;
            }
        }

        return nearest_core;
    }

    pub fn migrateBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, target_core: usize) !bool {
        const block_ptr = self.storage.getPtr(block_id);
        if (block_ptr == null) {
            return false;
        }
        var block = block_ptr.?;

        if (block.affinity_core) |old_core| {
            if (self.affinity_map.getPtr(old_core)) |old_set| {
                _ = old_set.remove(block_id);
            }
        }

        block.state = .migrating;
        block.affinity_core = target_core;

        var result = try self.affinity_map.getOrPut(target_core);
        if (!result.found_existing) {
            result.value_ptr.* = BlockIdSet.init(self.allocator);
        }
        try result.value_ptr.put(block_id, {});

        block.state = .allocated;
        return true;
    }

    const EvictionEntry = struct {
        id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        access_count: usize,
        last_access: i64,
        size: usize,
        entangled: bool,

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            if (a.access_count != b.access_count) {
                return a.access_count < b.access_count;
            }
            return a.last_access < b.last_access;
        }
    };

    pub fn evictLeastUsed(self: *ContentAddressableStorage, required_size: usize) !void {
        var blocks_to_sort = ArrayList(EvictionEntry).init(self.allocator);
        defer blocks_to_sort.deinit();

        var iter = self.storage.iterator();
        while (iter.next()) |entry| {
            const block = entry.value_ptr.*;
            try blocks_to_sort.append(.{
                .id = block.block_id,
                .access_count = block.access_count,
                .last_access = block.last_access_time,
                .size = block.size,
                .entangled = block.state == .entangled,
            });
        }

        std.mem.sort(EvictionEntry, blocks_to_sort.items, {}, EvictionEntry.lessThan);

        var freed: usize = 0;
        for (blocks_to_sort.items) |block_info| {
            if (freed >= required_size) {
                break;
            }
            if (!block_info.entangled) {
                self.removeBlock(block_info.id);
                freed += block_info.size;
            }
        }
    }

    fn removeBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) void {
        if (self.storage.fetchRemove(block_id)) |removed| {
            var block = removed.value;
            self.used_capacity -= block.size;

            if (self.content_index.getPtr(block.content_hash)) |block_set| {
                _ = block_set.remove(block_id);
                if (block_set.count() == 0) {
                    var removed_set = self.content_index.fetchRemove(block.content_hash);
                    if (removed_set) |rs| {
                        var set = rs.value;
                        set.deinit();
                    }
                }
            }

            if (block.affinity_core) |core_id| {
                if (self.affinity_map.getPtr(core_id)) |affinity_set| {
                    _ = affinity_set.remove(block_id);
                }
            }

            block.deinit();
        }
    }

    pub fn getStatistics(self: *const ContentAddressableStorage) StorageStatistics {
        var entangled_count: usize = 0;
        var iter = self.storage.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.state == .entangled) {
                entangled_count += 1;
            }
        }

        const utilization: f64 = if (self.total_capacity > 0)
            @as(f64, @floatFromInt(self.used_capacity)) / @as(f64, @floatFromInt(self.total_capacity))
        else
            0.0;

        return StorageStatistics{
            .total_blocks = self.storage.count(),
            .entangled_blocks = entangled_count,
            .used_capacity = self.used_capacity,
            .total_capacity = self.total_capacity,
            .utilization = utilization,
            .unique_contents = self.content_index.count(),
            .cores_with_affinity = self.affinity_map.count(),
        };
    }

    pub fn getBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?*MemoryBlock {
        return self.storage.getPtr(block_id);
    }

    pub fn containsBlock(self: *const ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return self.storage.contains(block_id);
    }
};

const TaskIdContext = struct {
    pub fn hash(_: @This(), key: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, b: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) bool {
        return std.mem.eql(u8, &a, &b);
    }
};

pub const DynamicTaskScheduler = struct {
    pending_tasks: ArrayList(TaskDescriptor),
    active_tasks: std.HashMap([ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, TaskDescriptor, TaskIdContext, std.hash_map.default_max_load_percentage),
    completed_tasks: ArrayList(TaskDescriptor),
    task_counter: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) DynamicTaskScheduler {
        return DynamicTaskScheduler{
            .pending_tasks = ArrayList(TaskDescriptor).init(allocator),
            .active_tasks = std.HashMap([ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, TaskDescriptor, TaskIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .completed_tasks = ArrayList(TaskDescriptor).init(allocator),
            .task_counter = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DynamicTaskScheduler) void {
        for (self.pending_tasks.items) |*task| {
            task.deinit();
        }
        self.pending_tasks.deinit();

        var active_iter = self.active_tasks.iterator();
        while (active_iter.next()) |entry| {
            var task = entry.value_ptr;
            task.deinit();
        }
        self.active_tasks.deinit();

        for (self.completed_tasks.items) |*task| {
            task.deinit();
        }
        self.completed_tasks.deinit();
    }

    fn generateTaskId(self: *DynamicTaskScheduler) [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        var buffer: [40]u8 = undefined;
        const task_str = std.fmt.bufPrint(&buffer, "task_{d}_{d}", .{ self.task_counter, std.time.milliTimestamp() }) catch "task_0_0";
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(task_str, &hash_out, .{});
        self.task_counter += 1;
        return hash_out;
    }

    pub fn submitTask(
        self: *DynamicTaskScheduler,
        priority: i32,
        data_dependencies: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        estimated_cycles: usize,
    ) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        const task_id = self.generateTaskId();
        var task = TaskDescriptor.init(self.allocator, task_id, priority, estimated_cycles);

        for (data_dependencies) |dep| {
            try task.addDependency(dep);
        }

        try self.pending_tasks.append(task);

        std.mem.sort(TaskDescriptor, self.pending_tasks.items, {}, struct {
            fn lessThan(_: void, a: TaskDescriptor, b: TaskDescriptor) bool {
                return a.priority < b.priority;
            }
        }.lessThan);

        return task_id;
    }

    pub fn scheduleTask(
        self: *DynamicTaskScheduler,
        cores: *AutoHashMap(usize, ProcessingCore),
        storage: *ContentAddressableStorage,
    ) !?struct { task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, core_id: usize } {
        if (self.pending_tasks.items.len == 0) {
            return null;
        }

        var task = self.pending_tasks.orderedRemove(0);

        var core_scores = AutoHashMap(usize, f64).init(self.allocator);
        defer core_scores.deinit();

        var core_iter = cores.iterator();
        while (core_iter.next()) |core_entry| {
            const core_id = core_entry.key_ptr.*;
            const core_info = core_entry.value_ptr.*;

            if (core_info.state == .power_gated) {
                continue;
            }

            var score: f64 = 0.0;
            for (task.data_dependencies.items) |dep_id| {
                if (storage.containsBlock(dep_id)) {
                    const nearest = storage.findNearestCore(dep_id, cores);
                    if (nearest) |nearest_core| {
                        if (nearest_core == core_id) {
                            score += 10.0;
                        } else {
                            score += 1.0;
                        }
                    }
                }
            }

            const workload = core_info.getWorkload();
            score += (1.0 - workload) * 5.0;

            try core_scores.put(core_id, score);
        }

        if (core_scores.count() == 0) {
            try self.pending_tasks.insert(0, task);
            return null;
        }

        var best_core: usize = 0;
        var best_score: f64 = -std.math.inf(f64);
        var score_iter = core_scores.iterator();
        while (score_iter.next()) |entry| {
            if (entry.value_ptr.* > best_score) {
                best_score = entry.value_ptr.*;
                best_core = entry.key_ptr.*;
            }
        }

        task.assigned_core = best_core;
        task.start_time = std.time.milliTimestamp();

        const task_id = task.task_id;
        try self.active_tasks.put(task_id, task);

        return .{ .task_id = task_id, .core_id = best_core };
    }

    pub fn completeTask(self: *DynamicTaskScheduler, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) !bool {
        if (self.active_tasks.fetchRemove(task_id)) |removed| {
            var task = removed.value;
            task.completion_status = true;
            task.end_time = std.time.milliTimestamp();
            try self.completed_tasks.append(task);
            return true;
        }
        return false;
    }

    pub fn getStatistics(self: *const DynamicTaskScheduler) SchedulerStatistics {
        var avg_completion_time: f64 = 0.0;
        if (self.completed_tasks.items.len > 0) {
            var total_time: i64 = 0;
            for (self.completed_tasks.items) |task| {
                total_time += task.getDuration();
            }
            avg_completion_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(self.completed_tasks.items.len));
        }

        return SchedulerStatistics{
            .pending_tasks = self.pending_tasks.items.len,
            .active_tasks = self.active_tasks.count(),
            .completed_tasks = self.completed_tasks.items.len,
            .average_completion_time = avg_completion_time,
        };
    }

    pub fn getActiveTask(self: *DynamicTaskScheduler, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) ?*TaskDescriptor {
        return self.active_tasks.getPtr(task_id);
    }
};

const FlowEdgeKey = struct {
    source: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    target: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
};

const FlowEdgeKeyContext = struct {
    pub fn hash(_: @This(), key: FlowEdgeKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&key.source);
        hasher.update(&key.target);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: FlowEdgeKey, b: FlowEdgeKey) bool {
        return std.mem.eql(u8, &a.source, &b.source) and std.mem.eql(u8, &a.target, &b.target);
    }
};

pub const AccessRecord = struct {
    timestamp: i64,
    core_id: usize,
};

pub const DataFlowAnalyzer = struct {
    flow_graph: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage),
    flow_weights: std.HashMap(FlowEdgeKey, f64, FlowEdgeKeyContext, std.hash_map.default_max_load_percentage),
    access_patterns: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, ArrayList(AccessRecord), BlockIdContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) DataFlowAnalyzer {
        return DataFlowAnalyzer{
            .flow_graph = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .flow_weights = std.HashMap(FlowEdgeKey, f64, FlowEdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .access_patterns = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, ArrayList(AccessRecord), BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DataFlowAnalyzer) void {
        var flow_iter = self.flow_graph.iterator();
        while (flow_iter.next()) |entry| {
            var set = entry.value_ptr;
            set.deinit();
        }
        self.flow_graph.deinit();

        self.flow_weights.deinit();

        var access_iter = self.access_patterns.iterator();
        while (access_iter.next()) |entry| {
            var list = entry.value_ptr;
            list.deinit();
        }
        self.access_patterns.deinit();
    }

    pub fn recordAccess(self: *DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, core_id: usize) !void {
        var result = try self.access_patterns.getOrPut(block_id);
        if (!result.found_existing) {
            result.value_ptr.* = ArrayList(AccessRecord).init(self.allocator);
        }
        try result.value_ptr.append(.{
            .timestamp = std.time.milliTimestamp(),
            .core_id = core_id,
        });
    }

    pub fn analyzeFlow(self: *const DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, allocator: Allocator) !AutoHashMap(usize, f64) {
        var core_affinities = AutoHashMap(usize, f64).init(allocator);

        const patterns_ptr = self.access_patterns.getPtr(block_id);
        if (patterns_ptr == null) {
            return core_affinities;
        }
        const accesses = patterns_ptr.?.items;

        if (accesses.len < 2) {
            return core_affinities;
        }

        var core_frequencies = AutoHashMap(usize, usize).init(allocator);
        defer core_frequencies.deinit();

        for (accesses) |access| {
            var entry = try core_frequencies.getOrPut(access.core_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }

        const total_accesses = @as(f64, @floatFromInt(accesses.len));
        var freq_iter = core_frequencies.iterator();
        while (freq_iter.next()) |entry| {
            const affinity = @as(f64, @floatFromInt(entry.value_ptr.*)) / total_accesses;
            try core_affinities.put(entry.key_ptr.*, affinity);
        }

        return core_affinities;
    }

    pub fn buildFlowGraph(self: *DataFlowAnalyzer, tasks: []const TaskDescriptor) !void {
        for (tasks) |task| {
            const deps = task.data_dependencies.items;
            var i: usize = 0;
            while (i < deps.len) : (i += 1) {
                const dep1 = deps[i];
                for (deps[i + 1 ..]) |dep2| {
                    var result1 = try self.flow_graph.getOrPut(dep1);
                    if (!result1.found_existing) {
                        result1.value_ptr.* = BlockIdSet.init(self.allocator);
                    }
                    try result1.value_ptr.put(dep2, {});

                    var result2 = try self.flow_graph.getOrPut(dep2);
                    if (!result2.found_existing) {
                        result2.value_ptr.* = BlockIdSet.init(self.allocator);
                    }
                    try result2.value_ptr.put(dep1, {});

                    var sorted_key: FlowEdgeKey = undefined;
                    if (std.mem.order(u8, &dep1, &dep2) == .lt) {
                        sorted_key = .{ .source = dep1, .target = dep2 };
                    } else {
                        sorted_key = .{ .source = dep2, .target = dep1 };
                    }

                    var weight_entry = try self.flow_weights.getOrPut(sorted_key);
                    if (!weight_entry.found_existing) {
                        weight_entry.value_ptr.* = 0.0;
                    }
                    weight_entry.value_ptr.* += 1.0;
                }
            }
        }
    }

    pub fn getCorrelatedBlocks(self: *const DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, threshold: f64, allocator: Allocator) !BlockIdSet {
        var correlated = BlockIdSet.init(allocator);

        const neighbors_ptr = self.flow_graph.getPtr(block_id);
        if (neighbors_ptr == null) {
            return correlated;
        }

        var neighbor_iter = neighbors_ptr.?.iterator();
        while (neighbor_iter.next()) |neighbor_entry| {
            const neighbor_id = neighbor_entry.key_ptr.*;

            var sorted_key: FlowEdgeKey = undefined;
            if (std.mem.order(u8, &block_id, &neighbor_id) == .lt) {
                sorted_key = .{ .source = block_id, .target = neighbor_id };
            } else {
                sorted_key = .{ .source = neighbor_id, .target = block_id };
            }

            if (self.flow_weights.get(sorted_key)) |weight| {
                if (weight >= threshold) {
                    try correlated.put(neighbor_id, {});
                }
            }
        }

        return correlated;
    }
};

pub const ChaosCoreKernel = struct {
    storage: ContentAddressableStorage,
    scheduler: DynamicTaskScheduler,
    flow_analyzer: DataFlowAnalyzer,
    cores: AutoHashMap(usize, ProcessingCore),
    cycle_count: usize,
    migration_count: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ChaosCoreKernel {
        return ChaosCoreKernel{
            .storage = ContentAddressableStorage.init(allocator),
            .scheduler = DynamicTaskScheduler.init(allocator),
            .flow_analyzer = DataFlowAnalyzer.init(allocator),
            .cores = AutoHashMap(usize, ProcessingCore).init(allocator),
            .cycle_count = 0,
            .migration_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ChaosCoreKernel) void {
        self.storage.deinit();
        self.scheduler.deinit();
        self.flow_analyzer.deinit();
        self.cores.deinit();
    }

    pub fn addCore(self: *ChaosCoreKernel, core_id: usize) !void {
        try self.cores.put(core_id, ProcessingCore.init(core_id));
    }

    pub fn removeCore(self: *ChaosCoreKernel, core_id: usize) bool {
        return self.cores.remove(core_id);
    }

    pub fn setCoreState(self: *ChaosCoreKernel, core_id: usize, state: CoreState) bool {
        if (self.cores.getPtr(core_id)) |core| {
            core.state = state;
            return true;
        }
        return false;
    }

    pub fn allocateMemory(self: *ChaosCoreKernel, data: []const u8, preferred_core: ?usize) ![ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        return self.storage.store(data, preferred_core);
    }

    pub fn readMemory(self: *ChaosCoreKernel, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
        return self.storage.retrieve(block_id);
    }

    pub fn createTask(
        self: *ChaosCoreKernel,
        priority: i32,
        data_dependencies: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        estimated_cycles: usize,
    ) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        const task_id = try self.scheduler.submitTask(priority, data_dependencies, estimated_cycles);

        if (self.scheduler.getActiveTask(task_id)) |task| {
            var task_slice = [_]TaskDescriptor{task.*};
            try self.flow_analyzer.buildFlowGraph(&task_slice);
        }

        return task_id;
    }

    pub fn finishTask(self: *ChaosCoreKernel, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) !bool {
        return self.scheduler.completeTask(task_id);
    }

    pub fn entangleData(self: *ChaosCoreKernel, block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !bool {
        const success = try self.storage.entangleBlocks(block_id1, block_id2);
        if (success) {
            var correlated = try self.flow_analyzer.getCorrelatedBlocks(block_id1, 0.5, self.allocator);
            defer correlated.deinit();

            var corr_iter = correlated.iterator();
            while (corr_iter.next()) |corr_entry| {
                const correlated_id = corr_entry.key_ptr.*;
                if (!std.mem.eql(u8, &correlated_id, &block_id2)) {
                    _ = try self.storage.entangleBlocks(block_id2, correlated_id);
                }
            }
        }
        return success;
    }

    pub fn getKernelStatistics(self: *const ChaosCoreKernel) KernelStatistics {
        return KernelStatistics{
            .cycle_count = self.cycle_count,
            .migration_count = self.migration_count,
            .storage = self.storage.getStatistics(),
            .scheduler = self.scheduler.getStatistics(),
        };
    }

    pub fn executeCycle(self: *ChaosCoreKernel) !void {
        self.cycle_count += 1;

        const scheduled = try self.scheduler.scheduleTask(&self.cores, &self.storage);
        if (scheduled) |sched_info| {
            if (self.scheduler.getActiveTask(sched_info.task_id)) |task| {
                for (task.data_dependencies.items) |dep_id| {
                    try self.flow_analyzer.recordAccess(dep_id, sched_info.core_id);
                }
            }
        }

        try self.optimizeDataPlacement();

        if (self.cycle_count % 100 == 0) {
            try self.balanceLoad();
        }
    }

    fn optimizeDataPlacement(self: *ChaosCoreKernel) !void {
        var blocks_to_migrate = ArrayList(struct { id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, target: usize }).init(self.allocator);
        defer blocks_to_migrate.deinit();

        var storage_iter = self.storage.storage.iterator();
        while (storage_iter.next()) |entry| {
            const block = entry.value_ptr.*;
            if (block.state == .migrating) {
                continue;
            }

            var affinities = try self.flow_analyzer.analyzeFlow(block.block_id, self.allocator);
            defer affinities.deinit();

            if (affinities.count() == 0) {
                continue;
            }

            var best_core: usize = 0;
            var best_affinity: f64 = -1.0;
            var affinity_iter = affinities.iterator();
            while (affinity_iter.next()) |aff_entry| {
                if (aff_entry.value_ptr.* > best_affinity) {
                    best_affinity = aff_entry.value_ptr.*;
                    best_core = aff_entry.key_ptr.*;
                }
            }

            const current_core = block.affinity_core orelse continue;
            if (current_core != best_core and best_affinity > 0.6) {
                try blocks_to_migrate.append(.{ .id = block.block_id, .target = best_core });
            }
        }

        for (blocks_to_migrate.items) |migration| {
            if (try self.storage.migrateBlock(migration.id, migration.target)) {
                self.migration_count += 1;
            }
        }
    }

    const CoreLoadEntry = struct {
        id: usize,
        load: f64,

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            return a.load < b.load;
        }
    };

    fn balanceLoad(self: *ChaosCoreKernel) !void {
        var core_loads = AutoHashMap(usize, f64).init(self.allocator);
        defer core_loads.deinit();

        var total_load: f64 = 0.0;
        var active_core_count: usize = 0;

        var core_iter = self.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.state == .power_gated) {
                continue;
            }

            const load = core.getWorkload();
            try core_loads.put(entry.key_ptr.*, load);
            total_load += load;
            active_core_count += 1;
        }

        if (active_core_count == 0) {
            return;
        }

        const avg_load = total_load / @as(f64, @floatFromInt(active_core_count));

        var overloaded = ArrayList(usize).init(self.allocator);
        defer overloaded.deinit();
        var underloaded = ArrayList(CoreLoadEntry).init(self.allocator);
        defer underloaded.deinit();

        var load_iter = core_loads.iterator();
        while (load_iter.next()) |entry| {
            if (entry.value_ptr.* > avg_load * 1.3) {
                try overloaded.append(entry.key_ptr.*);
            } else if (entry.value_ptr.* < avg_load * 0.7) {
                try underloaded.append(.{ .id = entry.key_ptr.*, .load = entry.value_ptr.* });
            }
        }

        if (overloaded.items.len == 0 or underloaded.items.len == 0) {
            return;
        }

        for (overloaded.items) |over_core| {
            if (self.storage.affinity_map.getPtr(over_core)) |block_set| {
                var blocks_list = ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8).init(self.allocator);
                defer blocks_list.deinit();

                var block_iter = block_set.iterator();
                while (block_iter.next()) |b_entry| {
                    try blocks_list.append(b_entry.key_ptr.*);
                }

                const migrate_count = @max(1, blocks_list.items.len / 4);
                for (blocks_list.items[0..migrate_count]) |block_id| {
                    if (underloaded.items.len > 0) {
                        std.mem.sort(CoreLoadEntry, underloaded.items, {}, CoreLoadEntry.lessThan);

                        const target_core = underloaded.items[0].id;
                        if (try self.storage.migrateBlock(block_id, target_core)) {
                            self.migration_count += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn selfOrganize(self: *ChaosCoreKernel) !void {
        var all_blocks = ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8).init(self.allocator);
        defer all_blocks.deinit();

        var storage_iter = self.storage.storage.iterator();
        while (storage_iter.next()) |entry| {
            try all_blocks.append(entry.key_ptr.*);
        }

        for (all_blocks.items) |block_id| {
            var correlated = try self.flow_analyzer.getCorrelatedBlocks(block_id, 0.4, self.allocator);
            defer correlated.deinit();

            var corr_iter = correlated.iterator();
            while (corr_iter.next()) |corr_entry| {
                const corr_id = corr_entry.key_ptr.*;
                if (self.storage.containsBlock(corr_id)) {
                    _ = try self.storage.entangleBlocks(block_id, corr_id);
                }
            }
        }

        try self.optimizeDataPlacement();
        try self.balanceLoad();
    }

    pub fn executeGraphOnKernel(self: *ChaosCoreKernel, graph: *SelfSimilarRelationalGraph) !std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, []const u8, BlockIdContext, std.hash_map.default_max_load_percentage) {
        var node_to_block = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, []const u8, BlockIdContext, std.hash_map.default_max_load_percentage).init(self.allocator);

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            const node = entry.value_ptr.*;
            const block_id = try self.allocateMemory(node.data, null);
            const node_id_copy = try self.allocator.dupe(u8, node_id);
            try node_to_block.put(block_id, node_id_copy);
        }

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge_key = entry.key_ptr.*;

            var source_block: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 = null;
            var target_block: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 = null;

            var map_iter = node_to_block.iterator();
            while (map_iter.next()) |map_entry| {
                if (std.mem.eql(u8, map_entry.value_ptr.*, edge_key.source)) {
                    source_block = map_entry.key_ptr.*;
                }
                if (std.mem.eql(u8, map_entry.value_ptr.*, edge_key.target)) {
                    target_block = map_entry.key_ptr.*;
                }
            }

            if (source_block != null and target_block != null) {
                _ = try self.entangleData(source_block.?, target_block.?);
            }
        }

        try self.selfOrganize();
        return node_to_block;
    }

    pub fn queryByContent(self: *ChaosCoreKernel, data: []const u8) ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        return self.storage.retrieveByContent(data);
    }

    pub fn getBlockLocation(self: *ChaosCoreKernel, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?usize {
        if (self.storage.getBlock(block_id)) |block| {
            return block.affinity_core;
        }
        return null;
    }

    pub fn formatBlockId(block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        var result: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        _ = std.fmt.bufPrint(&result, "{s}", .{std.fmt.fmtSliceHexLower(&block_id)}) catch {};
        return result;
    }

    pub fn formatTaskId(task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) [64]u8 {
        var result: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&result, "{s}", .{std.fmt.fmtSliceHexLower(&task_id)}) catch {};
        return result;
    }
};

test "MemoryBlockState toString and fromString" {
    const testing = std.testing;

    try testing.expectEqualStrings("free", MemoryBlockState.free.toString());
    try testing.expectEqualStrings("allocated", MemoryBlockState.allocated.toString());
    try testing.expectEqualStrings("entangled", MemoryBlockState.entangled.toString());
    try testing.expectEqualStrings("migrating", MemoryBlockState.migrating.toString());

    try testing.expectEqual(MemoryBlockState.free, MemoryBlockState.fromString("free").?);
    try testing.expectEqual(MemoryBlockState.allocated, MemoryBlockState.fromString("allocated").?);
    try testing.expect(MemoryBlockState.fromString("invalid") == null);
}

test "MemoryBlock initialization and access" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0xAB);
    var content_hash: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&content_hash, 0xCD);

    var block = try MemoryBlock.init(allocator, block_id, content_hash, "test data", 0);
    defer block.deinit();

    try testing.expectEqualStrings("test data", block.data);
    try testing.expectEqual(@as(usize, 9), block.size);
    try testing.expectEqual(MemoryBlockState.allocated, block.state);
    try testing.expectEqual(@as(?usize, 0), block.affinity_core);
    try testing.expectEqual(@as(usize, 1), block.access_count);
}

test "MemoryBlock entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id1, 0x11);
    var block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id2, 0x22);
    var content_hash: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&content_hash, 0x00);

    var block = try MemoryBlock.init(allocator, block_id1, content_hash, "data", null);
    defer block.deinit();

    try block.addEntangledBlock(block_id2);
    try testing.expect(block.hasEntanglement(block_id2));
    try testing.expectEqual(@as(usize, 1), block.entangledCount());
}

test "TaskDescriptor initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xFF);

    var task = TaskDescriptor.init(allocator, task_id, 5, 1000);
    defer task.deinit();

    try testing.expectEqual(@as(i32, 5), task.priority);
    try testing.expectEqual(@as(usize, 1000), task.estimated_cycles);
    try testing.expect(!task.completion_status);
}

test "ContentAddressableStorage store and retrieve" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    const block_id = try storage.store("hello world", 0);
    const retrieved = storage.retrieve(block_id);

    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("hello world", retrieved.?);
}

test "ContentAddressableStorage content deduplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    const block_id1 = try storage.store("duplicate", null);
    const block_id2 = try storage.store("duplicate", null);

    try testing.expect(std.mem.eql(u8, &block_id1, &block_id2));
    try testing.expectEqual(@as(usize, 1), storage.storage.count());
}

test "ContentAddressableStorage entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    const block_id1 = try storage.store("data1", null);
    const block_id2 = try storage.store("data2", null);

    const success = try storage.entangleBlocks(block_id1, block_id2);
    try testing.expect(success);

    const block1 = storage.getBlock(block_id1).?;
    const block2 = storage.getBlock(block_id2).?;

    try testing.expectEqual(MemoryBlockState.entangled, block1.state);
    try testing.expectEqual(MemoryBlockState.entangled, block2.state);
    try testing.expect(block1.hasEntanglement(block_id2));
    try testing.expect(block2.hasEntanglement(block_id1));
}

test "ContentAddressableStorage statistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    _ = try storage.store("test1", 0);
    _ = try storage.store("test2", 1);

    const stats = storage.getStatistics();
    try testing.expectEqual(@as(usize, 2), stats.total_blocks);
    try testing.expectEqual(@as(usize, 2), stats.unique_contents);
}

test "DynamicTaskScheduler submit and complete" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var scheduler = DynamicTaskScheduler.init(allocator);
    defer scheduler.deinit();

    var deps: [0][ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    const task_id = try scheduler.submitTask(1, &deps, 100);

    try testing.expectEqual(@as(usize, 1), scheduler.pending_tasks.items.len);

    var cores = AutoHashMap(usize, ProcessingCore).init(allocator);
    defer cores.deinit();
    try cores.put(0, ProcessingCore.init(0));

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    const scheduled = try scheduler.scheduleTask(&cores, &storage);
    try testing.expect(scheduled != null);
    try testing.expect(std.mem.eql(u8, &scheduled.?.task_id, &task_id));

    const completed = try scheduler.completeTask(task_id);
    try testing.expect(completed);
    try testing.expectEqual(@as(usize, 1), scheduler.completed_tasks.items.len);
}

test "DynamicTaskScheduler statistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var scheduler = DynamicTaskScheduler.init(allocator);
    defer scheduler.deinit();

    const stats = scheduler.getStatistics();
    try testing.expectEqual(@as(usize, 0), stats.pending_tasks);
    try testing.expectEqual(@as(usize, 0), stats.active_tasks);
    try testing.expectEqual(@as(usize, 0), stats.completed_tasks);
}

test "DataFlowAnalyzer record and analyze" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0x12);

    try analyzer.recordAccess(block_id, 0);
    try analyzer.recordAccess(block_id, 0);
    try analyzer.recordAccess(block_id, 1);

    var affinities = try analyzer.analyzeFlow(block_id, allocator);
    defer affinities.deinit();

    try testing.expect(affinities.count() == 2);

    const core0_affinity = affinities.get(0).?;
    try testing.expect(core0_affinity > 0.6);
}

test "DataFlowAnalyzer flow graph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xFF);

    var task = TaskDescriptor.init(allocator, task_id, 1, 100);
    defer task.deinit();

    var dep1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&dep1, 0x01);
    var dep2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&dep2, 0x02);

    try task.addDependency(dep1);
    try task.addDependency(dep2);

    var tasks = [_]TaskDescriptor{task};
    try analyzer.buildFlowGraph(&tasks);

    try testing.expect(analyzer.flow_graph.count() > 0);
}

test "ChaosCoreKernel initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 0), stats.cycle_count);
    try testing.expectEqual(@as(usize, 0), stats.migration_count);
}

test "ChaosCoreKernel memory operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);

    const block_id = try kernel.allocateMemory("test data", 0);
    const data = kernel.readMemory(block_id);

    try testing.expect(data != null);
    try testing.expectEqualStrings("test data", data.?);

    const location = kernel.getBlockLocation(block_id);
    try testing.expectEqual(@as(?usize, 0), location);
}

test "ChaosCoreKernel task management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);

    const block_id = try kernel.allocateMemory("dependency", 0);
    var deps = [_][ChaosCoreConfig.BLOCK_ID_SIZE]u8{block_id};

    const task_id = try kernel.createTask(1, &deps, 500);
    _ = task_id;

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 1), stats.scheduler.pending_tasks);
}

test "ChaosCoreKernel entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    const block_id1 = try kernel.allocateMemory("data1", null);
    const block_id2 = try kernel.allocateMemory("data2", null);

    const success = try kernel.entangleData(block_id1, block_id2);
    try testing.expect(success);

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 2), stats.storage.entangled_blocks);
}

test "ChaosCoreKernel query by content" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    const block_id = try kernel.allocateMemory("unique content", null);
    const found_id = kernel.queryByContent("unique content");

    try testing.expect(found_id != null);
    try testing.expect(std.mem.eql(u8, &block_id, &found_id.?));
}

test "ChaosCoreKernel cycle execution" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    try kernel.executeCycle();
    try kernel.executeCycle();

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 2), stats.cycle_count);
}

test "ProcessingCore workload calculation" {
    const testing = std.testing;

    var core = ProcessingCore.init(0);
    core.cycles_active = 75;
    core.cycles_idle = 25;

    const workload = core.getWorkload();
    try testing.expectApproxEqAbs(@as(f64, 0.75), workload, 0.001);
}

test "ChaosCoreKernel self organize" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    _ = try kernel.allocateMemory("block1", 0);
    _ = try kernel.allocateMemory("block2", 1);
    _ = try kernel.allocateMemory("block3", 0);

    try kernel.selfOrganize();

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 3), stats.storage.total_blocks);
}

test "ContentAddressableStorage migration" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    const block_id = try storage.store("migrate me", 0);
    const success = try storage.migrateBlock(block_id, 1);

    try testing.expect(success);

    const block = storage.getBlock(block_id).?;
    try testing.expectEqual(@as(?usize, 1), block.affinity_core);
}

test "ChaosCoreKernel format helpers" {
    const testing = std.testing;

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0xAB);

    const formatted = ChaosCoreKernel.formatBlockId(block_id);
    try testing.expectEqual(@as(usize, 32), formatted.len);

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xCD);

    const formatted_task = ChaosCoreKernel.formatTaskId(task_id);
    try testing.expectEqual(@as(usize, 64), formatted_task.len);
}
