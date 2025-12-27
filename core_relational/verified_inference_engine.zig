
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const zk = @import("zk_verification.zig");
const obf = @import("dataset_obfuscation.zig");
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;

pub const VerifiedInferenceEngine = struct {
    allocator: Allocator,
    commitment_scheme: *zk.CommitmentScheme,
    homomorphic_enc: *obf.HomomorphicEncryption,
    differential_privacy: *zk.DifferentialPrivacy,
    dataset_fingerprint: *obf.DatasetFingerprint,
    proof_of_correctness: *obf.ProofOfCorrectness,
    inference_proofs: ArrayList(*zk.ZKInferenceProof),
    model_hash: [32]u8,
    verification_count: u64,
    successful_verifications: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);

        self.* = Self{
            .allocator = allocator,
            .commitment_scheme = try zk.CommitmentScheme.init(allocator),
            .homomorphic_enc = try obf.HomomorphicEncryption.init(allocator),
            .differential_privacy = try zk.DifferentialPrivacy.init(allocator, 1.0, 1e-5, 1.0),
            .dataset_fingerprint = try obf.DatasetFingerprint.init(allocator),
            .proof_of_correctness = try obf.ProofOfCorrectness.init(allocator),
            .inference_proofs = ArrayList(*zk.ZKInferenceProof).init(allocator),
            .model_hash = undefined,
            .verification_count = 0,
            .successful_verifications = 0,
        };

        var model_hasher = Blake3.init(.{});
        const model_seed = "JAIDE_VERIFIED_MODEL_V40";
        model_hasher.update(model_seed);
        model_hasher.final(&self.model_hash);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.commitment_scheme.deinit();
        self.homomorphic_enc.deinit();
        self.differential_privacy.deinit();
        self.dataset_fingerprint.deinit();
        self.proof_of_correctness.deinit();

        for (self.inference_proofs.items) |proof| {
            proof.deinit();
        }
        self.inference_proofs.deinit();

        self.allocator.destroy(self);
    }

    pub fn performVerifiedInference(self: *Self, input: []const f32, output_buf: []f32) !void {
        if (input.len == 0 or output_buf.len == 0) {
            return error.InvalidInputOutput;
        }

        const input_commitment = try self.commitInput(input);

        var intermediate_1 = try self.allocator.alloc(f32, output_buf.len);
        defer self.allocator.free(intermediate_1);

        var i: usize = 0;
        while (i < intermediate_1.len and i < input.len) : (i += 1) {
            intermediate_1[i] = input[i] * 1.732;
        }
        while (i < intermediate_1.len) : (i += 1) {
            intermediate_1[i] = 0.0;
        }

        try self.proof_of_correctness.recordStep(
            1,
            input,
            intermediate_1,
            obf.ProofOfCorrectness.OperationType.MatrixMultiply,
        );

        var intermediate_2 = try self.allocator.alloc(f32, output_buf.len);
        defer self.allocator.free(intermediate_2);

        i = 0;
        while (i < intermediate_2.len) : (i += 1) {
            intermediate_2[i] = std.math.tanh(intermediate_1[i]);
        }

        try self.proof_of_correctness.recordStep(
            2,
            intermediate_1,
            intermediate_2,
            obf.ProofOfCorrectness.OperationType.Activation,
        );

        var sum: f64 = 0.0;
        for (intermediate_2) |val| {
            sum += @as(f64, val);
        }
        const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(intermediate_2.len)));

        i = 0;
        while (i < output_buf.len) : (i += 1) {
            output_buf[i] = intermediate_2[i] - mean;
        }

        try self.proof_of_correctness.recordStep(
            3,
            intermediate_2,
            output_buf,
            obf.ProofOfCorrectness.OperationType.Normalization,
        );

        for (output_buf) |*val| {
            const noisy = self.differential_privacy.addLaplaceNoise(@as(f64, val.*));
            val.* = @floatCast(noisy);
        }

        const output_commitment = try self.commitOutput(output_buf);

        const inference_proof = try zk.ZKInferenceProof.init(self.allocator);
        try inference_proof.proveInference(input, output_buf, self.model_hash);
        try self.inference_proofs.append(inference_proof);

        _ = input_commitment;
        _ = output_commitment;

        self.verification_count += 1;

        try self.proof_of_correctness.finalize();

        if (try self.verifyProofOfCorrectness()) {
            self.successful_verifications += 1;
        }
    }

    fn commitInput(self: *Self, input: []const f32) ![32]u8 {
        const input_bytes = std.mem.sliceAsBytes(input);
        return self.commitment_scheme.commit(input_bytes);
    }

    fn commitOutput(self: *Self, output: []const f32) ![32]u8 {
        const output_bytes = std.mem.sliceAsBytes(output);
        return self.commitment_scheme.commit(output_bytes);
    }

    fn verifyProofOfCorrectness(self: *Self) !bool {
        if (self.inference_proofs.items.len == 0) {
            return false;
        }

        const latest_proof = self.inference_proofs.items[self.inference_proofs.items.len - 1];
        return latest_proof.verify(self.model_hash);
    }

    pub fn getVerificationRate(self: *Self) f64 {
        if (self.verification_count == 0) {
            return 0.0;
        }
        return @as(f64, @floatFromInt(self.successful_verifications)) / @as(f64, @floatFromInt(self.verification_count));
    }

    pub fn proveDatasetIsolation(self: *Self, sample_data: []const u8) !bool {
        var hasher = Blake3.init(.{});
        hasher.update(sample_data);
        var sample_hash: [32]u8 = undefined;
        hasher.final(&sample_hash);

        const exists = self.dataset_fingerprint.checkSimilarity(sample_hash);

        return !exists;
    }

    pub fn generateZKProofForQuery(self: *Self, query: []const f32, response: []const f32) ![32]u8 {
        const proof = try zk.ZKInferenceProof.init(self.allocator);
        try proof.proveInference(query, response, self.model_hash);

        var proof_hasher = Blake3.init(.{});
        proof_hasher.update(&proof.input_commitment);
        proof_hasher.update(&proof.output_commitment);
        for (proof.computation_proof.items) |step| {
            proof_hasher.update(&step);
        }
        var proof_hash: [32]u8 = undefined;
        proof_hasher.final(&proof_hash);

        proof.deinit();

        return proof_hash;
    }
};

pub const BatchVerifier = struct {
    allocator: Allocator,
    batch_size: usize,
    accumulated_proofs: ArrayList([32]u8),
    batch_commitment: [32]u8,

    const Self = @This();

    pub fn init(allocator: Allocator, batch_size: usize) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .batch_size = batch_size,
            .accumulated_proofs = ArrayList([32]u8).init(allocator),
            .batch_commitment = undefined,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.accumulated_proofs.deinit();
        self.allocator.destroy(self);
    }

    pub fn addProof(self: *Self, proof_hash: [32]u8) !void {
        try self.accumulated_proofs.append(proof_hash);

        if (self.accumulated_proofs.items.len >= self.batch_size) {
            try self.finalizeBatch();
        }
    }

    fn finalizeBatch(self: *Self) !void {
        var hasher = Blake3.init(.{});

        for (self.accumulated_proofs.items) |proof| {
            hasher.update(&proof);
        }

        hasher.final(&self.batch_commitment);
        self.accumulated_proofs.clearRetainingCapacity();
    }

    pub fn verifyBatch(self: *Self, expected_commitment: [32]u8) bool {
        return std.mem.eql(u8, &self.batch_commitment, &expected_commitment);
    }
};
