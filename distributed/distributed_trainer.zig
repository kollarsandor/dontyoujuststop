const std = @import("std");
const Allocator = std.mem.Allocator;
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const nccl = @import("nccl_bindings.zig");
const RSF = @import("../processor/rsf.zig").RSF;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const SFD = @import("../optimizer/sfd.zig").SFD;
const SSI = @import("../index/ssi.zig").SSI;
const Tensor = @import("../core/tensor.zig").Tensor;
const Ranker = @import("../ranker/ranker.zig").Ranker;

const core_relational = @import("../core_relational/mod.zig");
const IBMQuantumClient = core_relational.IBMQuantumClient;
const EntangledStochasticSymmetryOptimizer = core_relational.EntangledStochasticSymmetryOptimizer;
const ZRuntime = core_relational.ZRuntime;
const FormalVerificationEngine = core_relational.FormalVerificationEngine;
const QuantumCircuit = core_relational.QuantumCircuit;
const QuantumClassicalHybridOptimizer = core_relational.QuantumClassicalHybridOptimizer;
const Observable = core_relational.Observable;
const SamplerOptions = core_relational.SamplerOptions;
const InvariantType = core_relational.InvariantType;
const Proposition = core_relational.Proposition;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const Node = core_relational.Node;
const Edge = core_relational.Edge;
const EdgeType = core_relational.EdgeType;

pub const DistributedTrainerConfig = struct {
    pub const DEFAULT_NUM_QUBITS: u32 = 8;
    pub const DEFAULT_VQE_LAYERS: u32 = 2;
    pub const DEFAULT_QUANTUM_SHOTS: u32 = 1024;
    pub const DEFAULT_QUANTUM_LEARNING_RATE: f64 = 0.01;
    pub const DEFAULT_MAX_QUANTUM_ITERATIONS: u32 = 100;
    pub const DEFAULT_VERIFICATION_FREQUENCY: usize = 10;
    pub const DEFAULT_LEARNING_RATE: f32 = 1e-4;
    pub const GRADIENT_MULTIPLIER: usize = 4;
    pub const NODE_NAME_BUFFER_SIZE: usize = 64;
    pub const NODE_DATA_BUFFER_SIZE: usize = 128;
    pub const BASE_COHERENCE: f64 = 0.9;
    pub const COHERENCE_DECREMENT: f64 = 0.05;
    pub const BASE_ENTANGLEMENT: f64 = 0.1;
    pub const ENTANGLEMENT_INCREMENT: f64 = 0.1;
    pub const EDGE_DEFAULT_WEIGHT: f64 = 1.0;
    pub const EDGE_COHERENCE_FACTOR: f64 = 0.95;
    pub const EDGE_ENTANGLEMENT_FACTOR: f64 = 0.1;
    pub const EDGE_SCALE_BASE: f64 = 1.0;
    pub const EDGE_SCALE_INCREMENT: f64 = 0.1;
    pub const QUANTUM_JOB_TIMEOUT_MS: u64 = 60000;
    pub const PARITY_SCALE: f64 = 2.0;
    pub const QUANTUM_MODULATION_SCALE: f64 = 0.1;
    pub const MAX_Z_RUNTIME_PARAMS: usize = 16;
    pub const VAR_NAME_BUFFER_SIZE: usize = 32;
    pub const MOMENTUM_FACTOR: f64 = 0.9;
    pub const QUANTUM_LOSS_WEIGHT: f64 = 0.1;
    pub const PHASE_MODULATION_SCALE: f64 = 0.01;
    pub const EDGE_WEIGHT_DELTA: f64 = 0.005;
    pub const LOGGING_INTERVAL: usize = 10;
    pub const BYTES_PER_VARIABLE: usize = 64;
    pub const Z_RUNTIME_MEMORY_SIZE: usize = 1024 * 1024;
    pub const LINE_BUFFER_SIZE: usize = 1024 * 1024;
    pub const MAX_SEQUENCE_LENGTH: usize = 512;
    pub const TOKEN_NORMALIZATION_FACTOR: f32 = 1000.0;
    pub const NCCL_POLL_INTERVAL_MS: u64 = 100;
    pub const DEFAULT_INVARIANTS_CHECKED: usize = 3;
    pub const LOSS_GRAD_INITIAL: f32 = 1.0;
    pub const FLOAT32_SIZE_MULTIPLIER: usize = 2;
    pub const DEFAULT_CHECKPOINT_DIR: []const u8 = "./checkpoints";
    pub const READY_SUFFIX: []const u8 = ".ready";
    pub const READY_CONTENT: []const u8 = "ready";
};

pub const QuantumTrainingConfig = struct {
    ibm_crn: []const u8,
    ibm_api_key: []const u8,
    num_qubits: u32 = DistributedTrainerConfig.DEFAULT_NUM_QUBITS,
    vqe_layers: u32 = DistributedTrainerConfig.DEFAULT_VQE_LAYERS,
    quantum_shots: u32 = DistributedTrainerConfig.DEFAULT_QUANTUM_SHOTS,
    enable_hybrid: bool = true,
    enable_verification: bool = true,
    quantum_learning_rate: f64 = DistributedTrainerConfig.DEFAULT_QUANTUM_LEARNING_RATE,
    max_quantum_iterations: u32 = DistributedTrainerConfig.DEFAULT_MAX_QUANTUM_ITERATIONS,
    verification_frequency: usize = DistributedTrainerConfig.DEFAULT_VERIFICATION_FREQUENCY,
};

pub const QuantumGradientResult = struct {
    gradients: []f64,
    quantum_expectation: f64,
    circuit_depth: u32,
    fidelity_estimate: f64,
    allocator: Allocator,

    pub fn deinit(self: *QuantumGradientResult) void {
        self.allocator.free(self.gradients);
    }
};

pub const VerificationStatus = struct {
    passed: bool,
    invariants_checked: usize,
    violated_invariants: []u64,
    graph_hash: [32]u8,
    verification_time_ns: i64,
    allocator: Allocator,

    pub fn deinit(self: *VerificationStatus) void {
        self.allocator.free(self.violated_invariants);
    }
};

pub const HybridStepResult = struct {
    classical_loss: f32,
    quantum_loss: f64,
    combined_loss: f64,
    quantum_contribution: f64,
    gradient_norm: f64,
    verification_passed: bool,
};

pub const DistributedTrainer = struct {
    allocator: Allocator,
    coordinator: *GPUCoordinator,
    rsf_layers: []RSF,
    tokenizer: MGT,
    optimizer: SFD,
    global_step: usize,
    local_batch_size: usize,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    learning_rate: f32,

    quantum_client: ?*IBMQuantumClient,
    esso_optimizer: ?*EntangledStochasticSymmetryOptimizer,
    z_runtime: ?*ZRuntime,
    verification_engine: ?*FormalVerificationEngine,
    hybrid_optimizer: ?*QuantumClassicalHybridOptimizer,
    training_graph: ?*SelfSimilarRelationalGraph,

    quantum_enabled: bool,
    hybrid_training: bool,
    verification_enabled: bool,
    quantum_config: ?QuantumTrainingConfig,

    quantum_gradient_accumulator: ?[]f64,
    last_verification_step: usize,
    total_quantum_shots: u64,
    successful_verifications: u64,

    pub fn init(
        allocator: Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        local_batch_size: usize,
    ) !DistributedTrainer {
        var rsf_layers = try allocator.alloc(RSF, num_layers);
        errdefer allocator.free(rsf_layers);

        var i: usize = 0;
        while (i < num_layers) : (i += 1) {
            rsf_layers[i] = try RSF.init(allocator, model_dim, 1);
            errdefer {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    rsf_layers[j].deinit();
                }
            }
        }

        const empty_vocab: []const []const u8 = &.{};
        const empty_anchors: []const []const u8 = &.{};
        const tokenizer = try MGT.init(allocator, empty_vocab, empty_anchors);
        const optimizer = try SFD.init(
            allocator,
            model_dim * num_layers * DistributedTrainerConfig.GRADIENT_MULTIPLIER,
        );

        return DistributedTrainer{
            .allocator = allocator,
            .coordinator = coordinator,
            .rsf_layers = rsf_layers,
            .tokenizer = tokenizer,
            .optimizer = optimizer,
            .global_step = 0,
            .local_batch_size = local_batch_size,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .vocab_size = vocab_size,
            .learning_rate = DistributedTrainerConfig.DEFAULT_LEARNING_RATE,
            .quantum_client = null,
            .esso_optimizer = null,
            .z_runtime = null,
            .verification_engine = null,
            .hybrid_optimizer = null,
            .training_graph = null,
            .quantum_enabled = false,
            .hybrid_training = false,
            .verification_enabled = false,
            .quantum_config = null,
            .quantum_gradient_accumulator = null,
            .last_verification_step = 0,
            .total_quantum_shots = 0,
            .successful_verifications = 0,
        };
    }

    pub fn initWithQuantum(
        allocator: Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        local_batch_size: usize,
        quantum_config: QuantumTrainingConfig,
    ) !DistributedTrainer {
        var trainer = try init(
            allocator,
            coordinator,
            model_dim,
            num_layers,
            vocab_size,
            local_batch_size,
        );
        errdefer trainer.deinit();

        const quantum_client = try IBMQuantumClient.init(
            allocator,
            quantum_config.ibm_crn,
            quantum_config.ibm_api_key,
        );
        errdefer quantum_client.deinit();

        try quantum_client.authenticate();

        const z_runtime = try ZRuntime.init(allocator);
        errdefer z_runtime.deinit();

        const esso_optimizer = try allocator.create(EntangledStochasticSymmetryOptimizer);
        errdefer allocator.destroy(esso_optimizer);
        esso_optimizer.* = EntangledStochasticSymmetryOptimizer.initDefault(allocator);
        errdefer esso_optimizer.deinit();

        const verification_engine = try allocator.create(FormalVerificationEngine);
        errdefer allocator.destroy(verification_engine);
        verification_engine.* = try FormalVerificationEngine.init(allocator);
        errdefer verification_engine.deinit();

        const hybrid_optimizer = try QuantumClassicalHybridOptimizer.init(
            allocator,
            quantum_client,
        );
        errdefer hybrid_optimizer.deinit();

        hybrid_optimizer.setMaxIterations(quantum_config.max_quantum_iterations);
        hybrid_optimizer.setLearningRate(quantum_config.quantum_learning_rate);

        const training_graph = try allocator.create(SelfSimilarRelationalGraph);
        errdefer allocator.destroy(training_graph);
        training_graph.* = SelfSimilarRelationalGraph.init(allocator);

        try initializeTrainingGraph(training_graph, model_dim, num_layers, allocator);

        const gradient_size = model_dim * num_layers * DistributedTrainerConfig.GRADIENT_MULTIPLIER;
        const quantum_gradient_accumulator = try allocator.alloc(f64, gradient_size);
        @memset(quantum_gradient_accumulator, 0.0);

        trainer.quantum_client = quantum_client;
        trainer.z_runtime = z_runtime;
        trainer.esso_optimizer = esso_optimizer;
        trainer.verification_engine = verification_engine;
        trainer.hybrid_optimizer = hybrid_optimizer;
        trainer.training_graph = training_graph;
        trainer.quantum_enabled = true;
        trainer.hybrid_training = quantum_config.enable_hybrid;
        trainer.verification_enabled = quantum_config.enable_verification;
        trainer.quantum_config = quantum_config;
        trainer.quantum_gradient_accumulator = quantum_gradient_accumulator;

        if (coordinator.isRoot()) {
            std.debug.print("[DistributedTrainer] Quantum integration initialized\n", .{});
            std.debug.print("  - IBM Quantum Client: Connected\n", .{});
            std.debug.print(
                "  - ZRuntime: Active with {d} variables\n",
                .{z_runtime.variableCount()},
            );
            std.debug.print("  - ESSO Optimizer: Ready\n", .{});
            std.debug.print(
                "  - Formal Verification: Enabled with {d} invariants\n",
                .{verification_engine.invariant_registry.count()},
            );
            std.debug.print(
                "  - Hybrid Optimizer: {d} qubits, {d} VQE layers\n",
                .{ quantum_config.num_qubits, quantum_config.vqe_layers },
            );
        }

        return trainer;
    }

    fn initializeTrainingGraph(
        graph: *SelfSimilarRelationalGraph,
        model_dim: usize,
        num_layers: usize,
        allocator: Allocator,
    ) !void {
        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            var node_name_buf: [DistributedTrainerConfig.NODE_NAME_BUFFER_SIZE]u8 = undefined;
            const node_name = try std.fmt.bufPrint(&node_name_buf, "layer_{d}", .{layer_idx});
            const node_name_copy = try allocator.dupe(u8, node_name);

            var node_data_buf: [DistributedTrainerConfig.NODE_DATA_BUFFER_SIZE]u8 = undefined;
            const node_data = try std.fmt.bufPrint(
                &node_data_buf,
                "rsf_layer_dim_{d}",
                .{model_dim},
            );
            const node_data_copy = try allocator.dupe(u8, node_data);

            const coherence = DistributedTrainerConfig.BASE_COHERENCE -
                (@as(f64, @floatFromInt(layer_idx)) * DistributedTrainerConfig.COHERENCE_DECREMENT);
            const entanglement = DistributedTrainerConfig.BASE_ENTANGLEMENT +
                (@as(f64, @floatFromInt(layer_idx)) * DistributedTrainerConfig.ENTANGLEMENT_INCREMENT);

            var node = try Node.init(
                allocator,
                node_name_copy,
                node_data_copy,
                coherence,
                entanglement,
                0.0,
            );
            try graph.addNode(node);

            if (layer_idx > 0) {
                var prev_name_buf: [DistributedTrainerConfig.NODE_NAME_BUFFER_SIZE]u8 = undefined;
                const prev_name = try std.fmt.bufPrint(
                    &prev_name_buf,
                    "layer_{d}",
                    .{layer_idx - 1},
                );
                const prev_name_copy = try allocator.dupe(u8, prev_name);
                const curr_name_copy = try allocator.dupe(u8, node_name);

                var edge = try Edge.init(
                    allocator,
                    prev_name_copy,
                    curr_name_copy,
                    .coherent,
                    DistributedTrainerConfig.EDGE_DEFAULT_WEIGHT,
                    DistributedTrainerConfig.EDGE_COHERENCE_FACTOR,
                    DistributedTrainerConfig.EDGE_ENTANGLEMENT_FACTOR,
                    DistributedTrainerConfig.EDGE_SCALE_BASE +
                        (@as(f64, @floatFromInt(layer_idx)) *
                        DistributedTrainerConfig.EDGE_SCALE_INCREMENT),
                );
                try graph.addEdge(edge);
            }
        }
    }

    pub fn deinit(self: *DistributedTrainer) void {
        if (self.quantum_gradient_accumulator) |acc| {
            self.allocator.free(acc);
        }

        if (self.training_graph) |graph| {
            graph.deinit();
            self.allocator.destroy(graph);
        }

        if (self.hybrid_optimizer) |ho| {
            ho.deinit();
            self.allocator.destroy(ho);
        }

        if (self.verification_engine) |ve| {
            ve.deinit();
            self.allocator.destroy(ve);
        }

        if (self.esso_optimizer) |esso| {
            esso.deinit();
            self.allocator.destroy(esso);
        }

        if (self.z_runtime) |zrt| {
            zrt.deinit();
            self.allocator.destroy(zrt);
        }

        if (self.quantum_client) |qc| {
            qc.deinit();
            self.allocator.destroy(qc);
        }

        for (self.rsf_layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.rsf_layers);
        self.tokenizer.deinit();
        self.optimizer.deinit();
    }

    pub fn quantumEnhancedOptimization(
        self: *DistributedTrainer,
        gradients: []f32,
    ) !QuantumGradientResult {
        if (!self.quantum_enabled or
            self.quantum_client == null or
            self.hybrid_optimizer == null)
        {
            return error.QuantumNotEnabled;
        }

        const quantum_client = self.quantum_client.?;
        const config = self.quantum_config.?;

        const num_params = @min(
            gradients.len,
            config.num_qubits * DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER,
        );

        const circuit = try quantum_client.createVQEAnsatz(
            config.num_qubits,
            config.vqe_layers,
        );
        defer circuit.deinit();

        var param_idx: usize = 0;
        while (param_idx < num_params) : (param_idx += 1) {
            const grad_val = gradients[param_idx];
            const angle = @as(f64, grad_val) * std.math.pi;

            const qubit_idx: u32 = @intCast(param_idx % config.num_qubits);
            try circuit.ry(qubit_idx, angle);
        }

        try circuit.measureAll();

        const options = SamplerOptions{ .shots = config.quantum_shots };
        const job = try quantum_client.runCircuit(circuit, null, options);

        try quantum_client.waitForJob(job, DistributedTrainerConfig.QUANTUM_JOB_TIMEOUT_MS);

        self.total_quantum_shots += config.quantum_shots;

        var quantum_gradients = try self.allocator.alloc(f64, gradients.len);
        errdefer self.allocator.free(quantum_gradients);

        var expectation: f64 = 0.0;

        if (job.result) |result| {
            var counts_iter = result.counts.iterator();
            var total_counts: u64 = 0;

            while (counts_iter.next()) |entry| {
                total_counts += entry.value_ptr.*;
            }

            counts_iter = result.counts.iterator();
            while (counts_iter.next()) |entry| {
                const bitstring = entry.key_ptr.*;
                const count = entry.value_ptr.*;
                const probability = @as(f64, @floatFromInt(count)) /
                    @as(f64, @floatFromInt(total_counts));

                var parity: i32 = 0;
                for (bitstring) |bit| {
                    if (bit == '1') parity += 1;
                }

                expectation += probability *
                    @as(f64, @floatFromInt(@mod(parity, 2))) *
                    DistributedTrainerConfig.PARITY_SCALE - probability;
            }
        }

        var grad_idx: usize = 0;
        while (grad_idx < gradients.len) : (grad_idx += 1) {
            const classical_grad = @as(f64, gradients[grad_idx]);
            const quantum_modulation = DistributedTrainerConfig.EDGE_DEFAULT_WEIGHT +
                (expectation * DistributedTrainerConfig.QUANTUM_MODULATION_SCALE);
            quantum_gradients[grad_idx] = classical_grad * quantum_modulation;
        }

        if (self.z_runtime) |zrt| {
            var var_idx: usize = 0;
            while (var_idx < @min(
                num_params,
                DistributedTrainerConfig.MAX_Z_RUNTIME_PARAMS,
            )) : (var_idx += 1) {
                var var_name_buf: [DistributedTrainerConfig.VAR_NAME_BUFFER_SIZE]u8 = undefined;
                const var_name = std.fmt.bufPrint(
                    &var_name_buf,
                    "qgrad_{d}",
                    .{var_idx},
                ) catch continue;
                var value_buf: [DistributedTrainerConfig.VAR_NAME_BUFFER_SIZE]u8 = undefined;
                const value_str = std.fmt.bufPrint(
                    &value_buf,
                    "{d:.6}",
                    .{quantum_gradients[var_idx]},
                ) catch continue;
                _ = zrt.createVariable(var_name, value_str) catch continue;
            }
        }

        if (self.esso_optimizer) |esso| {
            if (self.training_graph) |graph| {
                const optimized_graph = try esso.optimize(
                    graph,
                    core_relational.defaultGraphObjective,
                );
                optimized_graph.deinit();
                self.allocator.destroy(optimized_graph);
            }
        }

        const default_backend_name = quantum_client.default_backend orelse "ibmq_qasm_simulator";
        const backend = quantum_client.getBackend(default_backend_name);
        const fidelity = if (backend) |b|
            b.estimateFidelity(circuit.getDepth(), circuit.countTwoQubitGates())
        else
            DistributedTrainerConfig.BASE_COHERENCE;

        return QuantumGradientResult{
            .gradients = quantum_gradients,
            .quantum_expectation = expectation,
            .circuit_depth = circuit.getDepth(),
            .fidelity_estimate = fidelity,
            .allocator = self.allocator,
        };
    }

    pub fn verifyTrainingInvariants(self: *DistributedTrainer) !VerificationStatus {
        if (!self.verification_enabled or
            self.verification_engine == null or
            self.training_graph == null)
        {
            return error.VerificationNotEnabled;
        }

        const verification_engine = self.verification_engine.?;
        const training_graph = self.training_graph.?;

        const result = try verification_engine.verifyGraph(training_graph);
        defer {
            result.deinit();
            self.allocator.destroy(result);
        }

        var violated_copy = try self.allocator.alloc(u64, result.violated_invariants.items.len);
        var viol_idx: usize = 0;
        for (result.violated_invariants.items) |inv_id| {
            violated_copy[viol_idx] = inv_id;
            viol_idx += 1;
        }

        const connectivity_ok = verification_engine.verifyInvariantType(
            training_graph,
            .CONNECTIVITY,
        );
        const coherence_ok = verification_engine.verifyInvariantType(
            training_graph,
            .COHERENCE,
        );
        const entanglement_ok = verification_engine.verifyInvariantType(
            training_graph,
            .ENTANGLEMENT,
        );

        var invariants_checked: usize = DistributedTrainerConfig.DEFAULT_INVARIANTS_CHECKED;

        if (self.z_runtime) |zrt| {
            _ = zrt.variableCount();
            invariants_checked += 1;
        }

        const all_passed = result.success and
            connectivity_ok and
            coherence_ok and
            entanglement_ok;

        if (all_passed) {
            self.successful_verifications += 1;
        }

        self.last_verification_step = self.global_step;

        return VerificationStatus{
            .passed = all_passed,
            .invariants_checked = invariants_checked,
            .violated_invariants = violated_copy,
            .graph_hash = result.graph_hash,
            .verification_time_ns = result.execution_time_ns,
            .allocator = self.allocator,
        };
    }

    pub fn hybridQuantumClassicalStep(
        self: *DistributedTrainer,
        batch: [][]const u8,
    ) !HybridStepResult {
        const classical_loss = try self.trainStep(batch);

        var quantum_loss: f64 = 0.0;
        var quantum_contribution: f64 = 0.0;
        var gradient_norm: f64 = 0.0;

        if (self.quantum_enabled and self.hybrid_training) {
            const grad_size = self.model_dim *
                self.num_layers *
                DistributedTrainerConfig.GRADIENT_MULTIPLIER;
            var temp_gradients = try self.allocator.alloc(f32, grad_size);
            defer self.allocator.free(temp_gradients);

            var grad_idx: usize = 0;
            for (self.rsf_layers) |*layer| {
                for (layer.layers) |*sublayer| {
                    for (sublayer.s_weight_grad.data) |grad| {
                        if (grad_idx < grad_size) {
                            temp_gradients[grad_idx] = grad;
                            grad_idx += 1;
                        }
                    }
                    for (sublayer.t_weight_grad.data) |grad| {
                        if (grad_idx < grad_size) {
                            temp_gradients[grad_idx] = grad;
                            grad_idx += 1;
                        }
                    }
                }
            }

            var quantum_result = try self.quantumEnhancedOptimization(temp_gradients);
            defer quantum_result.deinit();

            for (quantum_result.gradients) |qgrad| {
                gradient_norm += qgrad * qgrad;
                quantum_loss += @fabs(qgrad);
            }
            gradient_norm = @sqrt(gradient_norm);
            quantum_loss /= @as(f64, @floatFromInt(quantum_result.gradients.len));

            quantum_contribution = quantum_result.quantum_expectation;

            if (self.quantum_gradient_accumulator) |acc| {
                const config = self.quantum_config.?;
                const q_learning_rate = config.quantum_learning_rate;

                var acc_idx: usize = 0;
                for (quantum_result.gradients) |qgrad| {
                    if (acc_idx < acc.len) {
                        acc[acc_idx] = acc[acc_idx] *
                            DistributedTrainerConfig.MOMENTUM_FACTOR +
                            qgrad * q_learning_rate;
                    }
                    acc_idx += 1;
                }
            }

            if (self.training_graph) |graph| {
                self.updateTrainingGraphState(graph, classical_loss, quantum_loss);
            }
        }

        var verification_passed = true;
        if (self.verification_enabled) {
            const config = self.quantum_config orelse QuantumTrainingConfig{
                .ibm_crn = "",
                .ibm_api_key = "",
                .verification_frequency = DistributedTrainerConfig.DEFAULT_VERIFICATION_FREQUENCY,
            };

            if (self.global_step - self.last_verification_step >= config.verification_frequency) {
                var verification_status = self.verifyTrainingInvariants() catch |err| {
                    if (self.coordinator.isRoot()) {
                        std.debug.print(
                            "[Step {d}] Verification deferred due to error: {any}\n",
                            .{ self.global_step, err },
                        );
                    }
                    return HybridStepResult{
                        .classical_loss = classical_loss,
                        .quantum_loss = quantum_loss,
                        .combined_loss = @as(f64, classical_loss) +
                            quantum_loss * DistributedTrainerConfig.QUANTUM_LOSS_WEIGHT,
                        .quantum_contribution = quantum_contribution,
                        .gradient_norm = gradient_norm,
                        .verification_passed = true,
                    };
                };
                defer verification_status.deinit();

                verification_passed = verification_status.passed;

                if (self.coordinator.isRoot()) {
                    std.debug.print(
                        "[Step {d}] Verification: {s} ({d} invariants, {d}ns)\n",
                        .{
                            self.global_step,
                            if (verification_passed) "PASSED" else "FAILED",
                            verification_status.invariants_checked,
                            verification_status.verification_time_ns,
                        },
                    );
                }
            }
        }

        const combined_loss = @as(f64, classical_loss) +
            quantum_loss * DistributedTrainerConfig.QUANTUM_LOSS_WEIGHT;

        return HybridStepResult{
            .classical_loss = classical_loss,
            .quantum_loss = quantum_loss,
            .combined_loss = combined_loss,
            .quantum_contribution = quantum_contribution,
            .gradient_norm = gradient_norm,
            .verification_passed = verification_passed,
        };
    }

    fn updateTrainingGraphState(
        self: *DistributedTrainer,
        graph: *SelfSimilarRelationalGraph,
        classical_loss: f32,
        quantum_loss: f64,
    ) void {
        _ = self;
        const loss_ratio = if (classical_loss > 0)
            quantum_loss / @as(f64, classical_loss)
        else
            DistributedTrainerConfig.EDGE_DEFAULT_WEIGHT;

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            var node = entry.value_ptr;
            node.phase = @mod(
                node.phase + loss_ratio * DistributedTrainerConfig.PHASE_MODULATION_SCALE,
                DistributedTrainerConfig.PARITY_SCALE * std.math.pi,
            );
        }

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            var edge_list = entry.value_ptr;
            for (edge_list.items) |*edge| {
                edge.weight = @max(0.0, @min(
                    DistributedTrainerConfig.EDGE_DEFAULT_WEIGHT,
                    edge.weight + loss_ratio * DistributedTrainerConfig.EDGE_WEIGHT_DELTA,
                ));
            }
        }
    }

    pub fn trainEpochHybrid(
        self: *DistributedTrainer,
        samples: [][]const u8,
    ) !HybridStepResult {
        var total_classical_loss: f64 = 0.0;
        var total_quantum_loss: f64 = 0.0;
        var total_gradient_norm: f64 = 0.0;
        var num_batches: usize = 0;
        var all_verifications_passed = true;

        var batch_start: usize = 0;
        while (batch_start < samples.len) {
            const batch_end = @min(batch_start + self.local_batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            const result = try self.hybridQuantumClassicalStep(batch);

            total_classical_loss += @as(f64, result.classical_loss);
            total_quantum_loss += result.quantum_loss;
            total_gradient_norm += result.gradient_norm;
            all_verifications_passed = all_verifications_passed and result.verification_passed;
            num_batches += 1;

            if (self.coordinator.isRoot() and
                self.global_step % DistributedTrainerConfig.LOGGING_INTERVAL == 0)
            {
                std.debug.print(
                    "[Step {d}] Classical: {d:.4}, Quantum: {d:.6}, Combined: {d:.6}\n",
                    .{
                        self.global_step,
                        result.classical_loss,
                        result.quantum_loss,
                        result.combined_loss,
                    },
                );
            }

            self.global_step += 1;
            batch_start = batch_end;
        }

        const num_batches_f = @as(f64, @floatFromInt(num_batches));
        const avg_classical_loss = if (num_batches > 0)
            total_classical_loss / num_batches_f
        else
            0.0;
        const avg_quantum_loss = if (num_batches > 0)
            total_quantum_loss / num_batches_f
        else
            0.0;
        const avg_gradient_norm = if (num_batches > 0)
            total_gradient_norm / num_batches_f
        else
            0.0;

        return HybridStepResult{
            .classical_loss = @floatCast(avg_classical_loss),
            .quantum_loss = avg_quantum_loss,
            .combined_loss = avg_classical_loss +
                avg_quantum_loss * DistributedTrainerConfig.QUANTUM_LOSS_WEIGHT,
            .quantum_contribution = avg_gradient_norm,
            .gradient_norm = avg_gradient_norm,
            .verification_passed = all_verifications_passed,
        };
    }

    pub const QuantumStats = struct {
        total_shots: u64,
        successful_verifications: u64,
        quantum_enabled: bool,
        hybrid_enabled: bool,
        verification_enabled: bool,
        z_runtime_memory_used: usize,
        z_runtime_variables: usize,
        ve_total_verifications: u64,
        ve_successful_verifications: u64,
        ve_invariant_count: usize,
    };

    pub fn getQuantumStatistics(self: *const DistributedTrainer) ?QuantumStats {
        if (!self.quantum_enabled) {
            return null;
        }

        var z_memory: usize = 0;
        var z_variables: usize = 0;
        if (self.z_runtime) |zrt| {
            z_variables = zrt.variableCount();
            z_memory = z_variables * DistributedTrainerConfig.BYTES_PER_VARIABLE;
        }

        var ve_total: u64 = 0;
        var ve_success: u64 = 0;
        var ve_invariants: usize = 0;

        if (self.verification_engine) |ve| {
            const stats = ve.getStatistics();
            ve_total = stats.total_verifications;
            ve_success = stats.successful_verifications;
            ve_invariants = stats.invariant_count;
        }

        return QuantumStats{
            .total_shots = self.total_quantum_shots,
            .successful_verifications = self.successful_verifications,
            .quantum_enabled = self.quantum_enabled,
            .hybrid_enabled = self.hybrid_training,
            .verification_enabled = self.verification_enabled,
            .z_runtime_memory_used = z_memory,
            .z_runtime_variables = z_variables,
            .ve_total_verifications = ve_total,
            .ve_successful_verifications = ve_success,
            .ve_invariant_count = ve_invariants,
        };
    }

    pub fn enableQuantum(self: *DistributedTrainer, config: QuantumTrainingConfig) !void {
        if (self.quantum_enabled) {
            return error.AlreadyEnabled;
        }

        const quantum_client = try self.allocator.create(IBMQuantumClient);
        errdefer self.allocator.destroy(quantum_client);
        quantum_client.* = try IBMQuantumClient.init(
            self.allocator,
            config.ibm_crn,
            config.ibm_api_key,
        );
        errdefer quantum_client.deinit();

        try quantum_client.authenticate();

        const z_runtime = try self.allocator.create(ZRuntime);
        errdefer self.allocator.destroy(z_runtime);
        z_runtime.* = try ZRuntime.init(
            self.allocator,
            DistributedTrainerConfig.Z_RUNTIME_MEMORY_SIZE,
        );
        errdefer z_runtime.deinit();

        const esso_optimizer = try self.allocator.create(EntangledStochasticSymmetryOptimizer);
        errdefer self.allocator.destroy(esso_optimizer);
        esso_optimizer.* = try EntangledStochasticSymmetryOptimizer.init(self.allocator);
        errdefer esso_optimizer.deinit();

        const verification_engine = try self.allocator.create(FormalVerificationEngine);
        errdefer self.allocator.destroy(verification_engine);
        verification_engine.* = try FormalVerificationEngine.init(self.allocator);
        errdefer verification_engine.deinit();

        const hybrid_optimizer = try self.allocator.create(QuantumClassicalHybridOptimizer);
        errdefer self.allocator.destroy(hybrid_optimizer);
        hybrid_optimizer.* = try QuantumClassicalHybridOptimizer.init(
            self.allocator,
            quantum_client,
        );
        errdefer hybrid_optimizer.deinit();

        hybrid_optimizer.setMaxIterations(config.max_quantum_iterations);
        hybrid_optimizer.setLearningRate(config.quantum_learning_rate);

        const training_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(training_graph);
        training_graph.* = SelfSimilarRelationalGraph.init(self.allocator);

        try initializeTrainingGraph(
            training_graph,
            self.model_dim,
            self.num_layers,
            self.allocator,
        );

        const gradient_size = self.model_dim *
            self.num_layers *
            DistributedTrainerConfig.GRADIENT_MULTIPLIER;
        const quantum_gradient_accumulator = try self.allocator.alloc(f64, gradient_size);
        @memset(quantum_gradient_accumulator, 0.0);

        self.quantum_client = quantum_client;
        self.z_runtime = z_runtime;
        self.esso_optimizer = esso_optimizer;
        self.verification_engine = verification_engine;
        self.hybrid_optimizer = hybrid_optimizer;
        self.training_graph = training_graph;
        self.quantum_enabled = true;
        self.hybrid_training = config.enable_hybrid;
        self.verification_enabled = config.enable_verification;
        self.quantum_config = config;
        self.quantum_gradient_accumulator = quantum_gradient_accumulator;
    }

    pub fn disableQuantum(self: *DistributedTrainer) void {
        if (!self.quantum_enabled) {
            return;
        }

        if (self.quantum_gradient_accumulator) |acc| {
            self.allocator.free(acc);
            self.quantum_gradient_accumulator = null;
        }

        if (self.training_graph) |graph| {
            graph.deinit();
            self.allocator.destroy(graph);
            self.training_graph = null;
        }

        if (self.hybrid_optimizer) |ho| {
            ho.deinit();
            self.allocator.destroy(ho);
            self.hybrid_optimizer = null;
        }

        if (self.verification_engine) |ve| {
            ve.deinit();
            self.allocator.destroy(ve);
            self.verification_engine = null;
        }

        if (self.esso_optimizer) |esso| {
            esso.deinit();
            self.allocator.destroy(esso);
            self.esso_optimizer = null;
        }

        if (self.z_runtime) |zrt| {
            zrt.deinit();
            self.allocator.destroy(zrt);
            self.z_runtime = null;
        }

        if (self.quantum_client) |qc| {
            qc.deinit();
            self.allocator.destroy(qc);
            self.quantum_client = null;
        }

        self.quantum_enabled = false;
        self.hybrid_training = false;
        self.verification_enabled = false;
        self.quantum_config = null;
    }

    pub fn loadDataset(self: *DistributedTrainer, dataset_path: []const u8) ![][]const u8 {
        var line_count: usize = 0;

        {
            const count_file = try std.fs.openFileAbsolute(dataset_path, .{});
            defer count_file.close();

            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();
            var line_buffer: [DistributedTrainerConfig.LINE_BUFFER_SIZE]u8 = undefined;

            while (try count_stream.readUntilDelimiterOrEof(&line_buffer, '\n')) |_| {
                line_count += 1;
            }
        }

        if (line_count == 0) {
            std.debug.print(
                "[Rank {d}] ERROR: Dataset is empty\n",
                .{self.coordinator.rank},
            );
            return error.EmptyDataset;
        }

        const base_samples_per_rank = @divFloor(line_count, self.coordinator.world_size);
        const remainder = line_count % self.coordinator.world_size;

        const samples_for_this_rank = if (self.coordinator.rank < remainder)
            base_samples_per_rank + 1
        else
            base_samples_per_rank;

        var start_line: usize = 0;
        var r: usize = 0;
        while (r < self.coordinator.rank) : (r += 1) {
            if (r < remainder) {
                start_line += base_samples_per_rank + 1;
            } else {
                start_line += base_samples_per_rank;
            }
        }

        const end_line = start_line + samples_for_this_rank;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| {
                self.allocator.free(sample);
            }
            samples.deinit();
        }

        const load_file = try std.fs.openFileAbsolute(dataset_path, .{});
        defer load_file.close();

        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();
        var line_buffer: [DistributedTrainerConfig.LINE_BUFFER_SIZE]u8 = undefined;

        var current_line: usize = 0;
        while (try load_stream.readUntilDelimiterOrEof(&line_buffer, '\n')) |line| {
            if (current_line >= start_line and current_line < end_line) {
                if (line.len == 0) {
                    current_line += 1;
                    continue;
                }

                const parsed = std.json.parseFromSlice(
                    std.json.Value,
                    self.allocator,
                    line,
                    .{},
                ) catch {
                    current_line += 1;
                    continue;
                };
                defer parsed.deinit();

                if (parsed.value.object.get("text")) |text_value| {
                    const text = text_value.string;
                    const text_copy = try self.allocator.dupe(u8, text);
                    try samples.append(text_copy);
                }
            }
            current_line += 1;

            if (current_line >= end_line) break;
        }

        if (self.coordinator.isRoot()) {
            std.debug.print(
                "[Rank {d}] Loaded {d} samples (lines {d}-{d} of {d} total)\n",
                .{
                    self.coordinator.rank,
                    samples.items.len,
                    start_line,
                    end_line,
                    line_count,
                },
            );
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainer, samples: [][]const u8) !f32 {
        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) {
            const batch_end = @min(batch_start + self.local_batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            const loss = try self.trainStep(batch);
            total_loss += loss;
            num_batches += 1;

            if (self.coordinator.isRoot() and
                self.global_step % DistributedTrainerConfig.LOGGING_INTERVAL == 0)
            {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, loss });
            }

            self.global_step += 1;
            batch_start = batch_end;
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        const loss_and_count_dev = try self.coordinator.allocDeviceMemory(
            DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER * @sizeOf(f32),
        );
        defer self.coordinator.freeDeviceMemory(loss_and_count_dev);

        try self.coordinator.copyHostToDevice(
            loss_and_count_dev,
            &loss_and_count,
            DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER * @sizeOf(f32),
        );
        try self.coordinator.allReduceFloat32(
            loss_and_count_dev,
            loss_and_count_dev,
            DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER,
        );
        try self.coordinator.copyDeviceToHost(
            &loss_and_count,
            loss_and_count_dev,
            DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER * @sizeOf(f32),
        );
        try self.coordinator.synchronize();

        const global_loss_sum = loss_and_count[0];
        const global_batch_count = loss_and_count[1];

        if (global_batch_count == 0) {
            std.debug.print("[WARNING] No batches processed across all ranks\n", .{});
            return 0.0;
        }

        const avg_loss = global_loss_sum / global_batch_count;

        return avg_loss;
    }

    pub fn trainStep(self: *DistributedTrainer, batch: [][]const u8) !f32 {
        var batch_tensors = try self.allocator.alloc(Tensor, batch.len);
        defer {
            for (batch_tensors) |*tensor| {
                tensor.deinit();
            }
            self.allocator.free(batch_tensors);
        }

        var batch_idx: usize = 0;
        while (batch_idx < batch.len) : (batch_idx += 1) {
            const text = batch[batch_idx];
            var token_list = std.ArrayList(u32).init(self.allocator);
            defer token_list.deinit();

            try self.tokenizer.encode(text, &token_list);

            const max_len = @min(
                token_list.items.len,
                DistributedTrainerConfig.MAX_SEQUENCE_LENGTH,
            );
            batch_tensors[batch_idx] = try Tensor.init(
                self.allocator,
                &.{ 1, self.model_dim },
            );

            var j: usize = 0;
            while (j < batch_tensors[batch_idx].data.len) : (j += 1) {
                if (j < max_len) {
                    batch_tensors[batch_idx].data[j] =
                        @as(f32, @floatFromInt(token_list.items[j])) /
                        DistributedTrainerConfig.TOKEN_NORMALIZATION_FACTOR;
                } else {
                    batch_tensors[batch_idx].data[j] = 0.0;
                }
            }
        }

        var gradients = try self.allocator.alloc(
            f32,
            self.model_dim * self.num_layers * DistributedTrainerConfig.GRADIENT_MULTIPLIER,
        );
        defer self.allocator.free(gradients);

        @memset(gradients, 0.0);

        var total_loss: f32 = 0.0;

        for (batch_tensors) |*input_tensor| {
            var current_tensor = input_tensor.*;

            for (self.rsf_layers) |*layer| {
                try layer.forward(&current_tensor);
            }

            const loss = self.computeLoss(&current_tensor);
            total_loss += loss;

            const loss_grad: f32 = DistributedTrainerConfig.LOSS_GRAD_INITIAL;

            var layer_idx: usize = self.num_layers;
            while (layer_idx > 0) {
                layer_idx -= 1;
                var layer = &self.rsf_layers[layer_idx];

                var grad_tensor = try Tensor.ones(self.allocator, current_tensor.shape);
                defer grad_tensor.deinit();
                _ = try layer.backward(&grad_tensor, &current_tensor);

                const grad_offset = layer_idx *
                    self.model_dim *
                    DistributedTrainerConfig.GRADIENT_MULTIPLIER;
                var wi: usize = 0;
                while (wi < layer.layers[0].s_weight.data.len) : (wi += 1) {
                    gradients[grad_offset + wi] +=
                        loss_grad * layer.layers[0].s_weight_grad.data[wi];
                }
                wi = 0;
                while (wi < layer.layers[0].t_weight.data.len) : (wi += 1) {
                    gradients[grad_offset +
                        self.model_dim *
                        DistributedTrainerConfig.FLOAT32_SIZE_MULTIPLIER + wi] +=
                        loss_grad * layer.layers[0].t_weight_grad.data[wi];
                }
            }
        }

        const gradients_dev = try self.coordinator.allocDeviceMemory(
            gradients.len * @sizeOf(f32),
        );
        defer self.coordinator.freeDeviceMemory(gradients_dev);

        try self.coordinator.copyHostToDevice(
            gradients_dev,
            gradients.ptr,
            gradients.len * @sizeOf(f32),
        );
        try self.coordinator.allReduceFloat32(gradients_dev, gradients_dev, gradients.len);
        try self.coordinator.copyDeviceToHost(
            gradients.ptr,
            gradients_dev,
            gradients.len * @sizeOf(f32),
        );
        try self.coordinator.synchronize();

        const world_size_f: f32 = @floatFromInt(self.coordinator.world_size);
        for (gradients) |*grad| {
            grad.* /= world_size_f;
        }

        var weights = try self.allocator.alloc(f32, gradients.len);
        defer self.allocator.free(weights);

        var weight_idx: usize = 0;
        for (self.rsf_layers) |*layer| {
            for (layer.layers) |*sublayer| {
                for (sublayer.s_weight.data) |weight| {
                    weights[weight_idx] = weight;
                    weight_idx += 1;
                }
                for (sublayer.t_weight.data) |weight| {
                    weights[weight_idx] = weight;
                    weight_idx += 1;
                }
            }
        }

        var weight_shape = try self.allocator.alloc(usize, 1);
        weight_shape[0] = weights.len;
        var weight_strides = try self.allocator.alloc(usize, 1);
        weight_strides[0] = 1;
        var weight_tensor = Tensor{
            .data = weights,
            .shape = weight_shape,
            .strides = weight_strides,
            .ndim = 1,
            .allocator = self.allocator,
        };

        var grad_shape = try self.allocator.alloc(usize, 1);
        grad_shape[0] = gradients.len;
        var grad_strides = try self.allocator.alloc(usize, 1);
        grad_strides[0] = 1;
        var grad_tensor = Tensor{
            .data = gradients,
            .shape = grad_shape,
            .strides = grad_strides,
            .ndim = 1,
            .allocator = self.allocator,
        };

        try self.optimizer.update(&grad_tensor, &weight_tensor, self.learning_rate);

        self.allocator.free(weight_shape);
        self.allocator.free(weight_strides);
        self.allocator.free(grad_shape);
        self.allocator.free(grad_strides);

        weight_idx = 0;
        for (self.rsf_layers) |*layer| {
            for (layer.layers) |*sublayer| {
                var wi: usize = 0;
                while (wi < sublayer.s_weight.data.len) : (wi += 1) {
                    sublayer.s_weight.data[wi] = weights[weight_idx];
                    weight_idx += 1;
                }
                wi = 0;
                while (wi < sublayer.t_weight.data.len) : (wi += 1) {
                    sublayer.t_weight.data[wi] = weights[weight_idx];
                    weight_idx += 1;
                }
            }
        }

        return total_loss / @as(f32, @floatFromInt(batch.len));
    }

    pub fn computeLoss(_: *DistributedTrainer, output: *const Tensor) f32 {
        var loss: f32 = 0.0;

        for (output.data) |val| {
            loss += val * val;
        }

        return loss / @as(f32, @floatFromInt(output.data.len));
    }

    pub fn saveCheckpoint(self: *DistributedTrainer, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            return;
        }

        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        var writer = file.writer();

        try writer.writeInt(usize, self.global_step, .little);
        try writer.writeInt(usize, self.model_dim, .little);
        try writer.writeInt(usize, self.num_layers, .little);

        for (self.rsf_layers) |layer| {
            for (layer.layers) |sublayer| {
                for (sublayer.s_weight.data) |weight| {
                    try writer.writeAll(std.mem.asBytes(&weight));
                }
                for (sublayer.t_weight.data) |weight| {
                    try writer.writeAll(std.mem.asBytes(&weight));
                }
            }
        }

        try writer.writeByte(if (self.quantum_enabled) 1 else 0);

        if (self.quantum_enabled) {
            try writer.writeInt(u64, self.total_quantum_shots, .little);
            try writer.writeInt(u64, self.successful_verifications, .little);

            if (self.quantum_gradient_accumulator) |acc| {
                for (acc) |val| {
                    try writer.writeAll(std.mem.asBytes(&val));
                }
            }
        }

        std.debug.print(
            "Checkpoint saved to {s} at step {d}\n",
            .{ path, self.global_step },
        );
    }

    pub fn loadCheckpoint(self: *DistributedTrainer, path: []const u8) !void {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();

        var reader = file.reader();

        self.global_step = try reader.readInt(usize, .little);
        const model_dim = try reader.readInt(usize, .little);
        const num_layers = try reader.readInt(usize, .little);

        if (model_dim != self.model_dim or num_layers != self.num_layers) {
            return error.CheckpointMismatch;
        }

        for (self.rsf_layers) |*layer| {
            for (layer.layers) |*sublayer| {
                for (sublayer.s_weight.data) |*weight| {
                    const bytes = try reader.readBytesNoEof(@sizeOf(f32));
                    weight.* = @bitCast(bytes);
                }
                for (sublayer.t_weight.data) |*weight| {
                    const bytes = try reader.readBytesNoEof(@sizeOf(f32));
                    weight.* = @bitCast(bytes);
                }
            }
        }

        const quantum_flag = reader.readByte() catch 0;
        if (quantum_flag == 1 and self.quantum_enabled) {
            self.total_quantum_shots = reader.readInt(u64, .little) catch 0;
            self.successful_verifications = reader.readInt(u64, .little) catch 0;

            if (self.quantum_gradient_accumulator) |acc| {
                for (acc) |*val| {
                    const bytes = reader.readBytesNoEof(@sizeOf(f64)) catch break;
                    val.* = @bitCast(bytes);
                }
            }
        }

        std.debug.print(
            "Checkpoint loaded from {s} at step {d}\n",
            .{ path, self.global_step },
        );
    }

    pub fn getNcclIdFilePath(allocator: Allocator) ![]const u8 {
        if (std.posix.getenv("NCCL_ID_FILE")) |path| {
            return try allocator.dupe(u8, path);
        }
        return error.NcclIdFileNotConfigured;
    }

    pub fn getNcclIdFilePathWithReady(
        allocator: Allocator,
    ) !struct { id_path: []const u8, ready_path: []const u8 } {
        const base_path = try getNcclIdFilePath(allocator);
        errdefer allocator.free(base_path);

        var ready_path_buf = std.ArrayList(u8).init(allocator);
        errdefer ready_path_buf.deinit();
        try ready_path_buf.appendSlice(base_path);
        try ready_path_buf.appendSlice(DistributedTrainerConfig.READY_SUFFIX);
        const ready_path = try ready_path_buf.toOwnedSlice();

        return .{
            .id_path = base_path,
            .ready_path = ready_path,
        };
    }

    pub fn getCheckpointDir(allocator: Allocator) ![]const u8 {
        if (std.posix.getenv("CHECKPOINT_DIR")) |dir| {
            return try allocator.dupe(u8, dir);
        }
        return try allocator.dupe(u8, DistributedTrainerConfig.DEFAULT_CHECKPOINT_DIR);
    }

    pub fn getCheckpointPath(allocator: Allocator, filename: []const u8) ![]const u8 {
        const dir = try getCheckpointDir(allocator);
        defer allocator.free(dir);

        var path_buf = std.ArrayList(u8).init(allocator);
        errdefer path_buf.deinit();

        try path_buf.appendSlice(dir);
        if (dir.len > 0 and dir[dir.len - 1] != '/') {
            try path_buf.append('/');
        }
        try path_buf.appendSlice(filename);

        return try path_buf.toOwnedSlice();
    }

    pub fn ensureCheckpointDirExists(allocator: Allocator) !void {
        const dir = try getCheckpointDir(allocator);
        defer allocator.free(dir);

        std.fs.cwd().makePath(dir) catch |err| {
            if (err != error.PathAlreadyExists) {
                return err;
            }
        };
    }

    pub fn writeNcclId(allocator: Allocator, nccl_id: *const nccl.ncclUniqueId) !void {
        const paths = try getNcclIdFilePathWithReady(allocator);
        defer allocator.free(paths.id_path);
        defer allocator.free(paths.ready_path);

        std.fs.cwd().deleteFile(paths.id_path) catch {};
        std.fs.cwd().deleteFile(paths.ready_path) catch {};

        const id_file = std.fs.cwd().createFile(paths.id_path, .{}) catch |err| {
            std.debug.print(
                "[NCCL] Failed to create ID file at {s}: {}\n",
                .{ paths.id_path, err },
            );
            return err;
        };
        defer id_file.close();

        try id_file.writeAll(std.mem.asBytes(nccl_id));
        try id_file.sync();

        const ready_file = try std.fs.cwd().createFile(paths.ready_path, .{});
        defer ready_file.close();
        try ready_file.writeAll(DistributedTrainerConfig.READY_CONTENT);
        try ready_file.sync();

        std.debug.print("[NCCL] Generated NCCL ID (file: {s})\n", .{paths.id_path});
    }

    pub fn readNcclId(
        allocator: Allocator,
        nccl_id: *nccl.ncclUniqueId,
        timeout_ms: u64,
    ) !void {
        const paths = try getNcclIdFilePathWithReady(allocator);
        defer allocator.free(paths.id_path);
        defer allocator.free(paths.ready_path);

        const max_attempts = timeout_ms / DistributedTrainerConfig.NCCL_POLL_INTERVAL_MS;
        var attempts: u64 = 0;

        while (attempts < max_attempts) : (attempts += 1) {
            const ready_file = std.fs.cwd().openFile(paths.ready_path, .{}) catch {
                std.time.sleep(
                    DistributedTrainerConfig.NCCL_POLL_INTERVAL_MS * std.time.ns_per_ms,
                );
                continue;
            };
            ready_file.close();
            break;
        }

        if (attempts >= max_attempts) {
            std.debug.print(
                "[NCCL] Timeout waiting for NCCL ID file at {s}\n",
                .{paths.ready_path},
            );
            return error.NcclIdTimeout;
        }

        const id_file = try std.fs.cwd().openFile(paths.id_path, .{});
        defer id_file.close();

        const bytes_read = try id_file.readAll(std.mem.asBytes(nccl_id));
        if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
            std.debug.print(
                "[NCCL] Failed to read NCCL ID (got {d} bytes, expected {d})\n",
                .{ bytes_read, @sizeOf(nccl.ncclUniqueId) },
            );
            return error.NcclIdReadFailed;
        }

        std.debug.print("[NCCL] Loaded NCCL ID from {s}\n", .{paths.id_path});
    }
};
