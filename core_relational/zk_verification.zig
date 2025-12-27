
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;
const Sha256 = crypto.hash.sha2.Sha256;
const Sha512 = crypto.hash.sha2.Sha512;

pub const CommitmentScheme = struct {
    allocator: Allocator,
    commitments: AutoHashMap([32]u8, Commitment),
    nonce_counter: u64,

    const Self = @This();

    pub const Commitment = struct {
        value_hash: [32]u8,
        nonce: [32]u8,
        timestamp: i64,
        blinding_factor: [32]u8,
    };

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .commitments = AutoHashMap([32]u8, Commitment).init(allocator),
            .nonce_counter = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.commitments.deinit();
        self.allocator.destroy(self);
    }

    pub fn commit(self: *Self, value: []const u8) ![32]u8 {
        var nonce: [32]u8 = undefined;
        crypto.random.bytes(&nonce);

        var blinding: [32]u8 = undefined;
        crypto.random.bytes(&blinding);

        var hasher = Blake3.init(.{});
        hasher.update(value);
        hasher.update(&nonce);
        hasher.update(&blinding);
        var commitment_hash: [32]u8 = undefined;
        hasher.final(&commitment_hash);

        var value_hasher = Sha256.init(.{});
        value_hasher.update(value);
        var value_hash: [32]u8 = undefined;
        value_hasher.final(&value_hash);

        const commitment = Commitment{
            .value_hash = value_hash,
            .nonce = nonce,
            .timestamp = std.time.milliTimestamp(),
            .blinding_factor = blinding,
        };

        try self.commitments.put(commitment_hash, commitment);
        self.nonce_counter += 1;

        return commitment_hash;
    }

    pub fn verify(self: *Self, commitment_hash: [32]u8, revealed_value: []const u8, revealed_nonce: [32]u8, revealed_blinding: [32]u8) !bool {
        const commitment = self.commitments.get(commitment_hash) orelse return false;

        var hasher = Blake3.init(.{});
        hasher.update(revealed_value);
        hasher.update(&revealed_nonce);
        hasher.update(&revealed_blinding);
        var computed_commitment: [32]u8 = undefined;
        hasher.final(&computed_commitment);

        if (!std.mem.eql(u8, &commitment_hash, &computed_commitment)) {
            return false;
        }

        var value_hasher = Sha256.init(.{});
        value_hasher.update(revealed_value);
        var computed_value_hash: [32]u8 = undefined;
        value_hasher.final(&computed_value_hash);

        return std.mem.eql(u8, &commitment.value_hash, &computed_value_hash);
    }
};

pub const RangeProof = struct {
    allocator: Allocator,
    min_value: i64,
    max_value: i64,
    proof_bits: ArrayList(ProofBit),

    const ProofBit = struct {
        commitment: [32]u8,
        opening: [32]u8,
        bit_value: u1,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, min: i64, max: i64) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .min_value = min,
            .max_value = max,
            .proof_bits = ArrayList(ProofBit).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.proof_bits.deinit();
        self.allocator.destroy(self);
    }

    pub fn prove(self: *Self, value: i64) !void {
        if (value < self.min_value or value > self.max_value) {
            return error.ValueOutOfRange;
        }

        const range = self.max_value - self.min_value;
        const bits_needed = @ctz(@as(u64, @intCast(range))) + 1;

        const normalized_value = value - self.min_value;

        var i: usize = 0;
        while (i < bits_needed) : (i += 1) {
            const bit: u1 = @intCast((normalized_value >> @intCast(i)) & 1);

            var nonce: [32]u8 = undefined;
            crypto.random.bytes(&nonce);

            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{bit});
            hasher.update(&nonce);
            var commitment: [32]u8 = undefined;
            hasher.final(&commitment);

            try self.proof_bits.append(ProofBit{
                .commitment = commitment,
                .opening = nonce,
                .bit_value = bit,
            });
        }
    }

    pub fn verify(self: *Self) !bool {
        var reconstructed_value: i64 = 0;
        var i: usize = 0;
        while (i < self.proof_bits.items.len) : (i += 1) {
            const bit_proof = self.proof_bits.items[i];
            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{bit_proof.bit_value});
            hasher.update(&bit_proof.opening);
            var computed_commitment: [32]u8 = undefined;
            hasher.final(&computed_commitment);

            if (!std.mem.eql(u8, &bit_proof.commitment, &computed_commitment)) {
                return false;
            }

            if (bit_proof.bit_value == 1) {
                reconstructed_value |= @as(i64, 1) << @intCast(i);
            }
        }

        const final_value = reconstructed_value + self.min_value;
        return final_value >= self.min_value and final_value <= self.max_value;
    }
};

pub const MembershipProof = struct {
    allocator: Allocator,
    merkle_root: [32]u8,
    path: ArrayList([32]u8),
    directions: ArrayList(bool),

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .merkle_root = undefined,
            .path = ArrayList([32]u8).init(allocator),
            .directions = ArrayList(bool).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.path.deinit();
        self.directions.deinit();
        self.allocator.destroy(self);
    }

    pub fn buildMerkleTree(self: *Self, elements: []const []const u8) ![32]u8 {
        if (elements.len == 0) return error.EmptySet;

        var current_level = ArrayList([32]u8).init(self.allocator);
        defer current_level.deinit();

        for (elements) |elem| {
            var hasher = Sha256.init(.{});
            hasher.update(elem);
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try current_level.append(hash);
        }

        while (current_level.items.len > 1) {
            var next_level = ArrayList([32]u8).init(self.allocator);
            defer next_level.deinit();

            var i: usize = 0;
            while (i < current_level.items.len) : (i += 2) {
                var hasher = Sha256.init(.{});
                hasher.update(&current_level.items[i]);

                if (i + 1 < current_level.items.len) {
                    hasher.update(&current_level.items[i + 1]);
                } else {
                    hasher.update(&current_level.items[i]);
                }

                var hash: [32]u8 = undefined;
                hasher.final(&hash);
                try next_level.append(hash);
            }

            current_level.clearRetainingCapacity();
            try current_level.appendSlice(next_level.items);
        }

        self.merkle_root = current_level.items[0];
        return self.merkle_root;
    }

    pub fn generateProof(self: *Self, elements: []const []const u8, index: usize) !void {
        if (index >= elements.len) return error.InvalidIndex;

        var current_level = ArrayList([32]u8).init(self.allocator);
        defer current_level.deinit();

        for (elements) |elem| {
            var hasher = Sha256.init(.{});
            hasher.update(elem);
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try current_level.append(hash);
        }

        var current_index = index;

        while (current_level.items.len > 1) {
            var sibling_index: usize = undefined;
            var direction: bool = undefined;

            if (current_index % 2 == 0) {
                sibling_index = current_index + 1;
                direction = false;
            } else {
                sibling_index = current_index - 1;
                direction = true;
            }

            if (sibling_index < current_level.items.len) {
                try self.path.append(current_level.items[sibling_index]);
            } else {
                try self.path.append(current_level.items[current_index]);
            }
            try self.directions.append(direction);

            var next_level = ArrayList([32]u8).init(self.allocator);
            defer next_level.deinit();

            var i: usize = 0;
            while (i < current_level.items.len) : (i += 2) {
                var hasher = Sha256.init(.{});
                hasher.update(&current_level.items[i]);

                if (i + 1 < current_level.items.len) {
                    hasher.update(&current_level.items[i + 1]);
                } else {
                    hasher.update(&current_level.items[i]);
                }

                var hash: [32]u8 = undefined;
                hasher.final(&hash);
                try next_level.append(hash);
            }

            current_level.clearRetainingCapacity();
            try current_level.appendSlice(next_level.items);
            current_index = current_index / 2;
        }
    }

    pub fn verify(self: *Self, element: []const u8) !bool {
        var hasher = Sha256.init(.{});
        hasher.update(element);
        var current_hash: [32]u8 = undefined;
        hasher.final(&current_hash);

        var i: usize = 0;
        while (i < self.path.items.len) : (i += 1) {
            const sibling = self.path.items[i];
            var next_hasher = Sha256.init(.{});

            if (self.directions.items[i]) {
                next_hasher.update(&sibling);
                next_hasher.update(&current_hash);
            } else {
                next_hasher.update(&current_hash);
                next_hasher.update(&sibling);
            }

            next_hasher.final(&current_hash);
        }

        return std.mem.eql(u8, &current_hash, &self.merkle_root);
    }
};

pub const SchnorrSignature = struct {
    challenge: [32]u8,
    response: [32]u8,

    const Self = @This();

    pub fn sign(allocator: Allocator, message: []const u8, private_key: [32]u8) !Self {
        _ = allocator;

        var k: [32]u8 = undefined;
        crypto.random.bytes(&k);

        var r_point: [32]u8 = undefined;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            r_point[i] = k[i] ^ private_key[i];
        }

        var challenge_hasher = Sha256.init(.{});
        challenge_hasher.update(&r_point);
        challenge_hasher.update(message);
        var challenge: [32]u8 = undefined;
        challenge_hasher.final(&challenge);

        var response: [32]u8 = undefined;
        i = 0;
        while (i < 32) : (i += 1) {
            response[i] = k[i] +% (challenge[i] *% private_key[i]);
        }

        return Self{
            .challenge = challenge,
            .response = response,
        };
    }

    pub fn verify(self: *Self, message: []const u8, public_key: [32]u8) bool {
        var r_point: [32]u8 = undefined;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            r_point[i] = self.response[i] -% (self.challenge[i] *% public_key[i]);
        }

        var challenge_hasher = Sha256.init(.{});
        challenge_hasher.update(&r_point);
        challenge_hasher.update(message);
        var computed_challenge: [32]u8 = undefined;
        challenge_hasher.final(&computed_challenge);

        return std.mem.eql(u8, &self.challenge, &computed_challenge);
    }
};

pub const DifferentialPrivacy = struct {
    allocator: Allocator,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    noise_scale: f64,

    const Self = @This();

    pub fn init(allocator: Allocator, epsilon: f64, delta: f64, sensitivity: f64) !*Self {
        const self = try allocator.create(Self);
        const noise_scale = sensitivity * @sqrt(2.0 * @log(1.25 / delta)) / epsilon;
        self.* = Self{
            .allocator = allocator,
            .epsilon = epsilon,
            .delta = delta,
            .sensitivity = sensitivity,
            .noise_scale = noise_scale,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn addNoise(self: *Self, value: f64) f64 {
        const rand1 = std.crypto.random.float(f64);
        const rand2 = std.crypto.random.float(f64);

        const z = @sqrt(-2.0 * @log(rand1)) * @cos(2.0 * std.math.pi * rand2);
        const noise = z * self.noise_scale;

        return value + noise;
    }

    pub fn addLaplaceNoise(self: *Self, value: f64) f64 {
        const u = std.crypto.random.float(f64) - 0.5;
        const b = self.sensitivity / self.epsilon;
        const abs_u = if (u < 0) -u else u;
        const noise = -b * @log(1.0 - 2.0 * abs_u) * if (u > 0) @as(f64, 1.0) else @as(f64, -1.0);
        return value + noise;
    }
};

pub const ZKInferenceProof = struct {
    allocator: Allocator,
    input_commitment: [32]u8,
    output_commitment: [32]u8,
    computation_proof: ArrayList([32]u8),
    timestamp: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .input_commitment = undefined,
            .output_commitment = undefined,
            .computation_proof = ArrayList([32]u8).init(allocator),
            .timestamp = std.time.milliTimestamp(),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.computation_proof.deinit();
        self.allocator.destroy(self);
    }

    pub fn proveInference(self: *Self, input: []const f32, output: []const f32, model_hash: [32]u8) !void {
        var input_hasher = Blake3.init(.{});
        for (input) |val| {
            const bytes = std.mem.asBytes(&val);
            input_hasher.update(bytes);
        }
        input_hasher.final(&self.input_commitment);

        var output_hasher = Blake3.init(.{});
        for (output) |val| {
            const bytes = std.mem.asBytes(&val);
            output_hasher.update(bytes);
        }
        output_hasher.final(&self.output_commitment);

        var step_hasher = Blake3.init(.{});
        step_hasher.update(&self.input_commitment);
        step_hasher.update(&model_hash);
        step_hasher.update(&self.output_commitment);
        var step_hash: [32]u8 = undefined;
        step_hasher.final(&step_hash);
        try self.computation_proof.append(step_hash);

        var i: usize = 0;
        while (i < 8) : (i += 1) {
            var intermediate_hasher = Blake3.init(.{});
            intermediate_hasher.update(&step_hash);
            const layer_index_bytes = std.mem.asBytes(&i);
            intermediate_hasher.update(layer_index_bytes);
            var intermediate_hash: [32]u8 = undefined;
            intermediate_hasher.final(&intermediate_hash);
            try self.computation_proof.append(intermediate_hash);
            step_hash = intermediate_hash;
        }
    }

    pub fn verify(self: *Self, model_hash: [32]u8) !bool {
        if (self.computation_proof.items.len < 1) {
            return false;
        }

        var first_hasher = Blake3.init(.{});
        first_hasher.update(&self.input_commitment);
        first_hasher.update(&model_hash);
        first_hasher.update(&self.output_commitment);
        var expected_first: [32]u8 = undefined;
        first_hasher.final(&expected_first);

        if (!std.mem.eql(u8, &self.computation_proof.items[0], &expected_first)) {
            return false;
        }

        var previous_hash = self.computation_proof.items[0];
        var i: usize = 1;
        while (i < self.computation_proof.items.len) : (i += 1) {
            var hasher = Blake3.init(.{});
            hasher.update(&previous_hash);
            const layer_index = i - 1;
            const layer_index_bytes = std.mem.asBytes(&layer_index);
            hasher.update(layer_index_bytes);
            var expected_hash: [32]u8 = undefined;
            hasher.final(&expected_hash);

            if (!std.mem.eql(u8, &self.computation_proof.items[i], &expected_hash)) {
                return false;
            }

            previous_hash = self.computation_proof.items[i];
        }

        return true;
    }
};

pub const SecureAggregation = struct {
    allocator: Allocator,
    participant_commitments: AutoHashMap(u64, [32]u8),
    aggregated_result: ?[]f64,
    threshold: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, threshold: usize) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .participant_commitments = AutoHashMap(u64, [32]u8).init(allocator),
            .aggregated_result = null,
            .threshold = threshold,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.aggregated_result) |result| {
            self.allocator.free(result);
        }
        self.participant_commitments.deinit();
        self.allocator.destroy(self);
    }

    pub fn commitParticipant(self: *Self, participant_id: u64, data: []const f64) ![32]u8 {
        var hasher = Blake3.init(.{});
        for (data) |val| {
            const bytes = std.mem.asBytes(&val);
            hasher.update(bytes);
        }
        var commitment: [32]u8 = undefined;
        hasher.final(&commitment);

        try self.participant_commitments.put(participant_id, commitment);
        return commitment;
    }

    pub fn aggregate(self: *Self, contributions: []const []const f64) !void {
        if (contributions.len < self.threshold) {
            return error.InsufficientParticipants;
        }

        const dim = contributions[0].len;
        const result = try self.allocator.alloc(f64, dim);
        for (result) |*val| {
            val.* = 0.0;
        }

        for (contributions) |contrib| {
            if (contrib.len != dim) {
                self.allocator.free(result);
                return error.DimensionMismatch;
            }
            var j: usize = 0;
            while (j < contrib.len) : (j += 1) {
                result[j] += contrib[j];
            }
        }

        const count: f64 = @floatFromInt(contributions.len);
        for (result) |*val| {
            val.* /= count;
        }

        if (self.aggregated_result) |old_result| {
            self.allocator.free(old_result);
        }
        self.aggregated_result = result;
    }

    pub fn getResult(self: *Self) ?[]const f64 {
        return self.aggregated_result;
    }
};
