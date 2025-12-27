
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;

pub const HomomorphicEncryption = struct {
    allocator: Allocator,
    public_key: [64]u8,
    private_key: [64]u8,
    modulus: u128,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);

        var pub_key: [64]u8 = undefined;
        crypto.random.bytes(&pub_key);

        var priv_key: [64]u8 = undefined;
        crypto.random.bytes(&priv_key);

        self.* = Self{
            .allocator = allocator,
            .public_key = pub_key,
            .private_key = priv_key,
            .modulus = 1_000_000_007,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn encrypt(self: *Self, plaintext: i64) !u128 {
        const abs_val: u64 = if (plaintext < 0) @intCast(-plaintext) else @intCast(plaintext);
        const pt: u128 = abs_val;
        const noise = crypto.random.int(u64);
        const ciphertext = (pt +% noise) % self.modulus;
        return ciphertext;
    }

    pub fn decrypt(self: *Self, ciphertext: u128) !i64 {
        _ = self;
        const plaintext: i64 = @intCast(ciphertext % 1000000);
        return plaintext;
    }

    pub fn add(self: *Self, c1: u128, c2: u128) u128 {
        return (c1 +% c2) % self.modulus;
    }

    pub fn multiply(self: *Self, c1: u128, scalar: i64) u128 {
        const abs_val: u64 = if (scalar < 0) @intCast(-scalar) else @intCast(scalar);
        const s: u128 = abs_val;
        return (c1 *% s) % self.modulus;
    }
};

pub const DatasetFingerprint = struct {
    allocator: Allocator,
    fingerprints: AutoHashMap([32]u8, FingerprintData),

    const FingerprintData = struct {
        sample_hash: [32]u8,
        encrypted_features: ArrayList(u128),
        timestamp: i64,
        access_count: u64,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .fingerprints = AutoHashMap([32]u8, FingerprintData).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        var iter = self.fingerprints.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.encrypted_features.deinit();
        }
        self.fingerprints.deinit();
        self.allocator.destroy(self);
    }

    pub fn addSample(self: *Self, sample: []const u8, features: []const i64, he: *HomomorphicEncryption) !void {
        var hasher = Blake3.init(.{});
        hasher.update(sample);
        var sample_hash: [32]u8 = undefined;
        hasher.final(&sample_hash);

        var encrypted_features = ArrayList(u128).init(self.allocator);
        for (features) |feat| {
            const encrypted = try he.encrypt(feat);
            try encrypted_features.append(encrypted);
        }

        const fingerprint = FingerprintData{
            .sample_hash = sample_hash,
            .encrypted_features = encrypted_features,
            .timestamp = std.time.milliTimestamp(),
            .access_count = 0,
        };

        try self.fingerprints.put(sample_hash, fingerprint);
    }

    pub fn checkSimilarity(self: *Self, query_hash: [32]u8) bool {
        return self.fingerprints.contains(query_hash);
    }
};

pub const SecureDataSampler = struct {
    allocator: Allocator,
    sample_pool: ArrayList(EncryptedSample),
    k_anonymity: usize,
    differential_privacy_budget: f64,

    const EncryptedSample = struct {
        id_hash: [32]u8,
        encrypted_data: ArrayList(u128),
        noise_signature: [32]u8,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, k_anonymity: usize, privacy_budget: f64) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .sample_pool = ArrayList(EncryptedSample).init(allocator),
            .k_anonymity = k_anonymity,
            .differential_privacy_budget = privacy_budget,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.sample_pool.items) |*sample| {
            sample.encrypted_data.deinit();
        }
        self.sample_pool.deinit();
        self.allocator.destroy(self);
    }

    pub fn addEncryptedSample(self: *Self, data: []const i64, he: *HomomorphicEncryption) !void {
        var id_bytes: [8]u8 = undefined;
        crypto.random.bytes(&id_bytes);

        var hasher = Blake3.init(.{});
        hasher.update(&id_bytes);
        var id_hash: [32]u8 = undefined;
        hasher.final(&id_hash);

        var encrypted_data = ArrayList(u128).init(self.allocator);
        for (data) |val| {
            const encrypted = try he.encrypt(val);
            try encrypted_data.append(encrypted);
        }

        var noise_sig: [32]u8 = undefined;
        crypto.random.bytes(&noise_sig);

        const sample = EncryptedSample{
            .id_hash = id_hash,
            .encrypted_data = encrypted_data,
            .noise_signature = noise_sig,
        };

        try self.sample_pool.append(sample);
    }

    pub fn sampleWithKAnonymity(self: *Self, count: usize) !ArrayList(usize) {
        if (count < self.k_anonymity) {
            return error.InsufficientKAnonymity;
        }

        var indices = ArrayList(usize).init(self.allocator);

        var i: usize = 0;
        while (i < count and i < self.sample_pool.items.len) : (i += 1) {
            try indices.append(i);
        }

        return indices;
    }
};

pub const ProofOfCorrectness = struct {
    allocator: Allocator,
    computation_trace: ArrayList(TraceStep),
    final_commitment: [32]u8,

    const TraceStep = struct {
        step_number: u64,
        input_hash: [32]u8,
        output_hash: [32]u8,
        operation_type: OperationType,
    };

    const OperationType = enum {
        MatrixMultiply,
        Activation,
        Normalization,
        Aggregation,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .computation_trace = ArrayList(TraceStep).init(allocator),
            .final_commitment = undefined,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.computation_trace.deinit();
        self.allocator.destroy(self);
    }

    pub fn recordStep(self: *Self, step_num: u64, input: []const f32, output: []const f32, op_type: OperationType) !void {
        var input_hasher = Blake3.init(.{});
        for (input) |val| {
            const bytes = std.mem.asBytes(&val);
            input_hasher.update(bytes);
        }
        var input_hash: [32]u8 = undefined;
        input_hasher.final(&input_hash);

        var output_hasher = Blake3.init(.{});
        for (output) |val| {
            const bytes = std.mem.asBytes(&val);
            output_hasher.update(bytes);
        }
        var output_hash: [32]u8 = undefined;
        output_hasher.final(&output_hash);

        const step = TraceStep{
            .step_number = step_num,
            .input_hash = input_hash,
            .output_hash = output_hash,
            .operation_type = op_type,
        };

        try self.computation_trace.append(step);
    }

    pub fn finalize(self: *Self) !void {
        var hasher = Blake3.init(.{});

        for (self.computation_trace.items) |step| {
            const step_num_bytes = std.mem.asBytes(&step.step_number);
            hasher.update(step_num_bytes);
            hasher.update(&step.input_hash);
            hasher.update(&step.output_hash);
        }

        hasher.final(&self.final_commitment);
    }

    pub fn verify(self: *Self, expected_commitment: [32]u8) bool {
        return std.mem.eql(u8, &self.final_commitment, &expected_commitment);
    }
};

pub const DatasetIsolation = struct {
    allocator: Allocator,
    isolation_barriers: ArrayList(IsolationBarrier),

    const IsolationBarrier = struct {
        dataset_id: u64,
        access_key: [32]u8,
        encrypted_metadata: ArrayList(u8),
        access_log: ArrayList(AccessRecord),
    };

    const AccessRecord = struct {
        timestamp: i64,
        operation_hash: [32]u8,
        success: bool,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .isolation_barriers = ArrayList(IsolationBarrier).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.isolation_barriers.items) |*barrier| {
            barrier.encrypted_metadata.deinit();
            barrier.access_log.deinit();
        }
        self.isolation_barriers.deinit();
        self.allocator.destroy(self);
    }

    pub fn createBarrier(self: *Self, dataset_id: u64) !u64 {
        var access_key: [32]u8 = undefined;
        crypto.random.bytes(&access_key);

        const barrier = IsolationBarrier{
            .dataset_id = dataset_id,
            .access_key = access_key,
            .encrypted_metadata = ArrayList(u8).init(self.allocator),
            .access_log = ArrayList(AccessRecord).init(self.allocator),
        };

        try self.isolation_barriers.append(barrier);
        return self.isolation_barriers.items.len - 1;
    }

    pub fn logAccess(self: *Self, barrier_idx: usize, operation: []const u8, success: bool) !void {
        if (barrier_idx >= self.isolation_barriers.items.len) {
            return error.InvalidBarrier;
        }

        var hasher = Blake3.init(.{});
        hasher.update(operation);
        var op_hash: [32]u8 = undefined;
        hasher.final(&op_hash);

        const record = AccessRecord{
            .timestamp = std.time.milliTimestamp(),
            .operation_hash = op_hash,
            .success = success,
        };

        try self.isolation_barriers.items[barrier_idx].access_log.append(record);
    }
};
