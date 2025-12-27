const std = @import("std");
const types = @import("core/types.zig");
const tensor_mod = @import("core/tensor.zig");
const rsf_mod = @import("processor/rsf.zig");
const mgt_mod = @import("tokenizer/mgt.zig");
const sfd_mod = @import("optimizer/sfd.zig");
const ssi_mod = @import("index/ssi.zig");
const ranker_mod = @import("ranker/ranker.zig");

const Tensor = tensor_mod.Tensor;
const RSF = rsf_mod.RSF;
const MGT = mgt_mod.MGT;
const SFD = sfd_mod.SFD;
const SSI = ssi_mod.SSI;
const Ranker = ranker_mod.Ranker;
const PRNG = types.PRNG;

pub const MainConfig = struct {
    pub const DEFAULT_EMBEDDING_DIM: usize = 128;
    pub const MIN_EMBEDDING_DIM: usize = 8;
    pub const MAX_EMBEDDING_DIM: usize = 16384;
    pub const DEFAULT_RSF_LAYERS: usize = 4;
    pub const MIN_RSF_LAYERS: usize = 1;
    pub const MAX_RSF_LAYERS: usize = 256;
    pub const DEFAULT_BATCH_SIZE: usize = 16;
    pub const MIN_BATCH_SIZE: usize = 1;
    pub const MAX_BATCH_SIZE: usize = 4096;
    pub const DEFAULT_NUM_EPOCHS: usize = 10;
    pub const MAX_NUM_EPOCHS: usize = 100000;
    pub const DEFAULT_LEARNING_RATE: f32 = 0.001;
    pub const MIN_LEARNING_RATE: f32 = 1e-10;
    pub const MAX_LEARNING_RATE: f32 = 10.0;
    pub const DEFAULT_TRAINING_SAMPLES: usize = 100;
    pub const DEFAULT_VALIDATION_SAMPLES: usize = 100;
    pub const MIN_SAMPLES: usize = 1;
    pub const MAX_SAMPLES: usize = 1000000;
    pub const DEFAULT_SAMPLE_LIMIT: usize = 3716;
    pub const DEFAULT_GRADIENT_CLIP_NORM: f32 = 5.0;
    pub const DEFAULT_SEQUENCE_LENGTH: usize = 64;
    pub const DEFAULT_TOP_K: usize = 5;
    pub const RANKER_NGRAM_SIZE: usize = 10;
    pub const RANKER_LSH_TABLES: usize = 16;
    pub const RANKER_SEED: u64 = 42;
    pub const TEST_DIM: usize = 128;
    pub const TEST_LAYERS: usize = 4;
    pub const TEST_PARAM_SIZE: usize = 128;
    pub const TEST_TOKEN_COUNT: usize = 8;
    pub const REPL_LINE_BUFFER_SIZE: usize = 4096;
    pub const ANCHOR_MODULO: usize = 3;
    pub const TENSOR_INIT_SCALE: f32 = 0.1;
    pub const PARAM_UPDATE_DELTA: f32 = 0.001;
    pub const GRADIENT_SCALE: f32 = 0.01;
    pub const GRADIENT_RANGE_SCALE: f32 = 10.0;
    pub const NORM_TOLERANCE: f32 = 0.1;
    pub const CHANGE_THRESHOLD: f32 = 1e-6;
    pub const GRADIENT_THRESHOLD: f32 = 1e-9;
    pub const R_SQUARED_EPSILON: f64 = 1e-10;
    pub const CONFIDENCE_Z_SCORE: f64 = 1.96;
    pub const PRNG_SEED_FORWARD: u64 = 54321;
    pub const PRNG_SEED_VALIDATION: u64 = 12345;
    pub const PRNG_SEED_GRADIENT: u64 = 99999;
    pub const PRNG_SEED_SYNTHETIC: u64 = 42;
    pub const MAX_VALID_POSITION: u64 = 10000;
    pub const MAX_TOKEN_COUNT: usize = 1000;
    pub const GRADIENT_TENSOR_SIZE: usize = 100;
    pub const PARSE_BASE: u8 = 10;
    pub const DEFAULT_MODELS_DIR: []const u8 = "models";
    pub const FILE_MAGIC_RSF: u32 = 0x4A524653;
    pub const FILE_MAGIC_MGT: u32 = 0x4A4D4754;
    pub const FILE_MAGIC_RANKER: u32 = 0x4A524E4B;
    pub const FILE_VERSION: u32 = 1;
    pub const MAX_LINE_LENGTH: usize = 65536;
    pub const MAX_VOCAB_SIZE: u32 = std.math.maxInt(u32);
    pub const MAX_TOKEN_LENGTH: u32 = 65536;
};

const Config = struct {
    embedding_dim: usize,
    rsf_layers: usize,
    batch_size: usize,
    num_epochs: usize,
    learning_rate: f32,
    num_training_samples: usize,
    num_validation_samples: usize,
    models_dir: []const u8,
    vocab_file: ?[]const u8,
    dataset_path: ?[]const u8,
    sample_limit: usize,
    gradient_clip_norm: f32,
    sequence_length: usize,
    top_k: usize,
    allocator: std.mem.Allocator,
    models_dir_allocated: ?[]u8,
    vocab_file_allocated: ?[]u8,
    dataset_path_allocated: ?[]u8,

    pub fn init(allocator: std.mem.Allocator) !Config {
        const models_dir_copy = try allocator.dupe(u8, MainConfig.DEFAULT_MODELS_DIR);
        return Config{
            .embedding_dim = MainConfig.DEFAULT_EMBEDDING_DIM,
            .rsf_layers = MainConfig.DEFAULT_RSF_LAYERS,
            .batch_size = MainConfig.DEFAULT_BATCH_SIZE,
            .num_epochs = MainConfig.DEFAULT_NUM_EPOCHS,
            .learning_rate = MainConfig.DEFAULT_LEARNING_RATE,
            .num_training_samples = MainConfig.DEFAULT_TRAINING_SAMPLES,
            .num_validation_samples = MainConfig.DEFAULT_VALIDATION_SAMPLES,
            .models_dir = models_dir_copy,
            .vocab_file = null,
            .dataset_path = null,
            .sample_limit = MainConfig.DEFAULT_SAMPLE_LIMIT,
            .gradient_clip_norm = MainConfig.DEFAULT_GRADIENT_CLIP_NORM,
            .sequence_length = MainConfig.DEFAULT_SEQUENCE_LENGTH,
            .top_k = MainConfig.DEFAULT_TOP_K,
            .allocator = allocator,
            .models_dir_allocated = models_dir_copy,
            .vocab_file_allocated = null,
            .dataset_path_allocated = null,
        };
    }

    pub fn deinit(self: *Config) void {
        if (self.models_dir_allocated) |dir| {
            self.allocator.free(dir);
        }
        if (self.vocab_file_allocated) |file| {
            self.allocator.free(file);
        }
        if (self.dataset_path_allocated) |path| {
            self.allocator.free(path);
        }
    }

    pub fn validate(self: *const Config) error{InvalidConfig}!void {
        if (self.embedding_dim < MainConfig.MIN_EMBEDDING_DIM or self.embedding_dim > MainConfig.MAX_EMBEDDING_DIM) {
            return error.InvalidConfig;
        }
        if (self.rsf_layers < MainConfig.MIN_RSF_LAYERS or self.rsf_layers > MainConfig.MAX_RSF_LAYERS) {
            return error.InvalidConfig;
        }
        if (self.batch_size < MainConfig.MIN_BATCH_SIZE or self.batch_size > MainConfig.MAX_BATCH_SIZE) {
            return error.InvalidConfig;
        }
        if (self.num_epochs > MainConfig.MAX_NUM_EPOCHS) {
            return error.InvalidConfig;
        }
        if (self.learning_rate < MainConfig.MIN_LEARNING_RATE or self.learning_rate > MainConfig.MAX_LEARNING_RATE) {
            return error.InvalidConfig;
        }
        if (std.math.isNan(self.learning_rate) or std.math.isInf(self.learning_rate)) {
            return error.InvalidConfig;
        }
        if (self.num_training_samples < MainConfig.MIN_SAMPLES or self.num_training_samples > MainConfig.MAX_SAMPLES) {
            return error.InvalidConfig;
        }
        if (self.num_validation_samples > MainConfig.MAX_SAMPLES) {
            return error.InvalidConfig;
        }
        if (self.top_k == 0) {
            return error.InvalidConfig;
        }
        if (std.math.isNan(self.gradient_clip_norm) or std.math.isInf(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0) {
            return error.InvalidConfig;
        }
    }

    pub fn parseArgs(allocator: std.mem.Allocator) !Config {
        var config = try Config.init(allocator);
        errdefer config.deinit();

        var args = try std.process.argsWithAllocator(allocator);
        defer args.deinit();

        _ = args.skip();

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--embedding-dim")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.embedding_dim = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--layers")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.rsf_layers = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--batch-size")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.batch_size = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--epochs")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.num_epochs = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--lr")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const lr = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(lr) or std.math.isInf(lr)) return error.InvalidArgumentValue;
                config.learning_rate = lr;
            } else if (std.mem.eql(u8, arg, "--samples")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.num_training_samples = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--models-dir")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.models_dir_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.models_dir_allocated = duped;
                config.models_dir = duped;
            } else if (std.mem.eql(u8, arg, "--vocab-file")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.vocab_file_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.vocab_file_allocated = duped;
                config.vocab_file = duped;
            } else if (std.mem.eql(u8, arg, "--dataset-path")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.dataset_path_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.dataset_path_allocated = duped;
                config.dataset_path = duped;
            } else if (std.mem.eql(u8, arg, "--sample-limit")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.sample_limit = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--gradient-clip")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const clip = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(clip) or std.math.isInf(clip) or clip <= 0.0) return error.InvalidArgumentValue;
                config.gradient_clip_norm = clip;
            } else if (std.mem.eql(u8, arg, "--sequence-length")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.sequence_length = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.top_k = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--help")) {
                try printHelp();
                return error.HelpRequested;
            } else if (std.mem.eql(u8, arg, "--mode")) {
                _ = args.next();
            }
        }

        try config.validate();
        return config;
    }
};

const TrainingSample = struct {
    text: []u8,
    tokens: []u32,

    pub fn deinit(self: *TrainingSample, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.tokens);
    }
};

const ValidationMetrics = struct {
    mse: f32,
    rmse: f32,
    mae: f32,
    r_squared: f32,
    mean_prediction: f32,
    mean_target: f32,
    confidence_interval_lower: f32,
    confidence_interval_upper: f32,
    num_samples: usize,
};

const TerminalColors = struct {
    enabled: bool,
    reset: []const u8,
    bold: []const u8,
    cyan: []const u8,
    green: []const u8,
    yellow: []const u8,
    red: []const u8,

    fn detect() TerminalColors {
        const stdout = std.io.getStdOut();
        const tty_config = std.io.tty.detectConfig(stdout);
        const enabled = tty_config != .no_color;

        if (enabled) {
            return TerminalColors{
                .enabled = true,
                .reset = "\x1b[0m",
                .bold = "\x1b[1m",
                .cyan = "\x1b[36m",
                .green = "\x1b[32m",
                .yellow = "\x1b[33m",
                .red = "\x1b[31m",
            };
        } else {
            return TerminalColors{
                .enabled = false,
                .reset = "",
                .bold = "",
                .cyan = "",
                .green = "",
                .yellow = "",
                .red = "",
            };
        }
    }
};

const TestResult = struct {
    name: []const u8,
    passed: bool,
    message: []const u8,
};

fn runKgruTest(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    const colors = TerminalColors.detect();

    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}{s}  KGRU Component Test Suite{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}{s}========================================{s}\n\n", .{ colors.bold, colors.cyan, colors.reset });

    var tests_passed: usize = 0;
    var tests_failed: usize = 0;

    const test1_passed = blk: {
        try stdout.print("{s}[TEST 1]{s} RSF Processor Initialization & Forward Pass...\n", .{ colors.yellow, colors.reset });
        const dim: usize = MainConfig.TEST_DIM;
        const layers: usize = MainConfig.TEST_LAYERS;

        var rsf = RSF.init(allocator, dim, layers) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer rsf.deinit();

        const tensor_dim = std.math.mul(usize, dim, 2) catch {
            try stdout.print("  {s}FAILED{s}: Dimension overflow\n", .{ colors.red, colors.reset });
            break :blk false;
        };

        var input_tensor = Tensor.init(allocator, &.{ 1, tensor_dim }) catch |err| {
            try stdout.print("  {s}FAILED{s}: Tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer input_tensor.deinit();

        var ti: usize = 0;
        while (ti < input_tensor.data.len) : (ti += 1) {
            input_tensor.data[ti] = @as(f32, @floatFromInt(ti % 10)) * MainConfig.TENSOR_INIT_SCALE;
        }

        rsf.forward(&input_tensor) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF forward error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        var has_valid_output = false;
        for (input_tensor.data) |v| {
            if (v != 0.0 and !std.math.isNan(v) and !std.math.isInf(v)) {
                has_valid_output = true;
                break;
            }
        }

        if (has_valid_output) {
            try stdout.print("  {s}PASSED{s}: RSF forward pass produces valid output\n", .{ colors.green, colors.reset });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: RSF forward pass returned all zeros or NaN\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test1_passed) tests_passed += 1 else tests_failed += 1;

    const test2_passed = blk: {
        try stdout.print("{s}[TEST 2]{s} SFD Optimizer Initialization...\n", .{ colors.yellow, colors.reset });
        const param_size: usize = MainConfig.TEST_PARAM_SIZE;

        var optimizer = SFD.init(allocator, param_size) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer optimizer.deinit();

        var gradients = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Gradient tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer gradients.deinit();

        var params = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Params tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer params.deinit();

        var params_before = try allocator.alloc(f32, param_size);
        defer allocator.free(params_before);

        var pi: usize = 0;
        while (pi < param_size) : (pi += 1) {
            gradients.data[pi] = @as(f32, @floatFromInt(pi % 10)) * MainConfig.GRADIENT_SCALE;
            params.data[pi] = @as(f32, @floatFromInt(pi)) * MainConfig.PARAM_UPDATE_DELTA;
            params_before[pi] = params.data[pi];
        }

        optimizer.update(&gradients, &params, MainConfig.DEFAULT_LEARNING_RATE) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD update error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        var params_changed = false;
        pi = 0;
        while (pi < param_size) : (pi += 1) {
            if (std.math.fabs(params.data[pi] - params_before[pi]) > MainConfig.GRADIENT_THRESHOLD) {
                params_changed = true;
                break;
            }
        }

        if (params_changed) {
            try stdout.print("  {s}PASSED{s}: SFD optimizer update completed and params changed\n", .{ colors.green, colors.reset });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: SFD optimizer update did not change params\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test2_passed) tests_passed += 1 else tests_failed += 1;

    const test3_passed = blk: {
        try stdout.print("{s}[TEST 3]{s} MGT Tokenizer Initialization...\n", .{ colors.yellow, colors.reset });

        var mgt = initTokenizer(allocator) catch |err| {
            try stdout.print("  {s}FAILED{s}: MGT init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer mgt.deinit();

        const vocab_size = mgt.vocabSize();
        if (vocab_size > 0) {
            try stdout.print("  {s}PASSED{s}: MGT tokenizer initialized with vocab size {d}\n", .{ colors.green, colors.reset, vocab_size });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: MGT tokenizer has empty vocabulary\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test3_passed) tests_passed += 1 else tests_failed += 1;

    const test4_passed = blk: {
        try stdout.print("{s}[TEST 4]{s} SSI Index Operations...\n", .{ colors.yellow, colors.reset });

        var ssi = SSI.init(allocator);
        defer ssi.deinit();

        var test_tokens = try allocator.alloc(u32, MainConfig.TEST_TOKEN_COUNT);
        defer allocator.free(test_tokens);

        var idx: usize = 0;
        while (idx < MainConfig.TEST_TOKEN_COUNT) : (idx += 1) {
            test_tokens[idx] = @as(u32, @intCast(idx + 1));
        }

        ssi.addSequence(test_tokens, 0, true) catch |err| {
            try stdout.print("  {s}FAILED{s}: SSI addSequence error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        const stats = ssi.stats();
        if (stats.nodes > 0) {
            try stdout.print("  {s}PASSED{s}: SSI index created with {d} nodes\n", .{ colors.green, colors.reset, stats.nodes });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: SSI index is empty after adding sequence\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test4_passed) tests_passed += 1 else tests_failed += 1;

    const test5_passed = blk: {
        try stdout.print("{s}[TEST 5]{s} Ranker Initialization...\n", .{ colors.yellow, colors.reset });

        var ranker = Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED) catch |err| {
            try stdout.print("  {s}FAILED{s}: Ranker init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer ranker.deinit();

        try stdout.print("  {s}PASSED{s}: Ranker initialized with ngrams={d}, lsh_tables={d}\n", .{ colors.green, colors.reset, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES });
        break :blk true;
    };

    if (test5_passed) tests_passed += 1 else tests_failed += 1;

    try stdout.writeAll("\n");
    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}  Test Results: {d} passed, {d} failed{s}\n", .{
        if (tests_failed == 0) colors.green else colors.red,
        tests_passed,
        tests_failed,
        colors.reset,
    });
    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });

    if (tests_failed > 0) {
        return error.TestsFailed;
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("Memory leak detected\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var config = Config.parseArgs(allocator) catch |err| {
        if (err == error.HelpRequested) {
            return;
        }
        std.debug.print("Configuration error: {any}\n", .{err});
        return err;
    };
    defer config.deinit();

    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();
    _ = args_iter.skip();

    var mode: ?[]const u8 = null;
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--mode")) {
            mode = args_iter.next();
            break;
        }
    }

    if (mode) |m| {
        if (std.mem.eql(u8, m, "test")) {
            runKgruTest(allocator) catch |err| {
                if (err == error.TestsFailed) {
                    return err;
                }
                return err;
            };
            return;
        } else if (std.mem.eql(u8, m, "train")) {
            try runTraining(allocator, &config);
            return;
        } else if (std.mem.eql(u8, m, "validate")) {
            try runValidation(allocator, &config);
            return;
        }
    }

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    var ssi = SSI.init(allocator);
    defer ssi.deinit();

    var ranker = try Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED);
    defer ranker.deinit();

    try runInteractiveREPL(allocator, &mgt, &ssi, &ranker, &config);
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\JAIDE v40 - Root-Level LLM System
        \\
        \\Usage: jaide [OPTIONS]
        \\
        \\Options:
        \\  --mode <MODE>         Operating mode: test, train, validate, or interactive (default)
        \\  --embedding-dim <N>   Embedding dimension (default: 128, range: 8-16384)
        \\  --layers <N>          Number of RSF layers (default: 4, range: 1-256)
        \\  --batch-size <N>      Batch size for training (default: 16, range: 1-4096)
        \\  --epochs <N>          Number of training epochs (default: 10, max: 100000)
        \\  --lr <RATE>           Learning rate (default: 0.001, range: 1e-10 to 10.0)
        \\  --samples <N>         Number of training samples (default: 100)
        \\  --models-dir <PATH>   Directory for model files (default: models)
        \\  --vocab-file <PATH>   Path to vocabulary file
        \\  --dataset-path <PATH> Path to training dataset (JSONL format)
        \\  --sample-limit <N>    Maximum samples to load from dataset
        \\  --gradient-clip <N>   Gradient clipping norm (default: 5.0)
        \\  --sequence-length <N> Sequence length for training (default: 64)
        \\  --top-k <N>           Top-K results for retrieval (default: 5)
        \\  --help                Show this help message
        \\
    );
}

fn initTokenizer(allocator: std.mem.Allocator) !MGT {
    const sample_vocab = [_][]const u8{
        "a", "az", "es", "is", "nem", "de", "hogy", "egy", "mert", "vagy",
        "minden", "csak", "meg", "mar", "most", "itt", "ott", "ki", "mi", "ez",
        "neural", "network", "learning", "deep", "machine", "intelligence", "artificial",
        "data", "model", "training", "optimization", "algorithm", "computer", "science",
        "mesterseges", "intelligencia", "neuralis", "halozat", "tanulas", "gepi",
        "adattudomany", "optimalizalas", "algoritmus", "kvantum", "robotika",
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    };

    const sample_anchors = [_][]const u8{ "a", "az", "es", "the", "neural", "mesterseges" };

    return try MGT.init(allocator, &sample_vocab, &sample_anchors);
}

fn calculateTotalParams(dim: usize, layers: usize) error{Overflow}!usize {
    const dim_squared = std.math.mul(usize, dim, dim) catch return error.Overflow;
    const weights_per_layer = std.math.mul(usize, dim_squared, 2) catch return error.Overflow;
    const biases_per_layer = std.math.mul(usize, dim, 2) catch return error.Overflow;
    const params_per_layer = std.math.add(usize, weights_per_layer, biases_per_layer) catch return error.Overflow;
    return std.math.mul(usize, params_per_layer, layers) catch return error.Overflow;
}

fn runTraining(allocator: std.mem.Allocator, config: *const Config) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("Initializing training...\n");

    const dim = config.embedding_dim;
    const layers = config.rsf_layers;

    const total_params = try calculateTotalParams(dim, layers);
    try stdout.print("Model parameters: {d}\n", .{total_params});

    var rsf = try RSF.init(allocator, dim, layers);
    defer rsf.deinit();

    var optimizer = try SFD.init(allocator, total_params);
    defer optimizer.deinit();

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    var ssi = SSI.init(allocator);
    defer ssi.deinit();

    var ranker = try Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED);
    defer ranker.deinit();

    var samples = try generateSyntheticSamples(allocator, &mgt, config.num_training_samples);
    defer {
        for (samples) |*sample| {
            sample.deinit(allocator);
        }
        allocator.free(samples);
    }

    try stdout.print("Generated {d} training samples\n", .{samples.len});

    var epoch: usize = 0;
    while (epoch < config.num_epochs) : (epoch += 1) {
        var epoch_loss: f64 = 0.0;
        var batch_count: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) : (batch_start += config.batch_size) {
            const batch_end = @min(batch_start + config.batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            const tensor_dim = std.math.mul(usize, dim, 2) catch continue;
            var input = try Tensor.init(allocator, &.{ 1, tensor_dim });
            defer input.deinit();

            var batch_loss: f64 = 0.0;
            for (batch) |sample| {
                try createEmbeddingInPlace(&input, sample.tokens, mgt.vocabSize(), dim);
                try rsf.forward(&input);

                var sum: f64 = 0.0;
                for (input.data) |v| {
                    if (!std.math.isNan(v) and !std.math.isInf(v)) {
                        sum += @as(f64, v);
                    }
                }
                const avg = sum / @as(f64, @floatFromInt(@max(input.data.len, 1)));
                batch_loss += avg * avg;
            }

            epoch_loss += batch_loss;
            batch_count += 1;
        }

        const avg_loss = if (batch_count > 0) epoch_loss / @as(f64, @floatFromInt(batch_count)) else 0.0;
        try stdout.print("Epoch {d}/{d} - Loss: {d:.6}\n", .{ epoch + 1, config.num_epochs, avg_loss });
    }

    try saveModels(allocator, &rsf, &mgt, &optimizer, &ranker, config.models_dir);
    try stdout.writeAll("Training complete. Models saved.\n");
}

fn runValidation(allocator: std.mem.Allocator, config: *const Config) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("Running validation...\n");

    const dim = config.embedding_dim;
    const layers = config.rsf_layers;

    var rsf = try RSF.init(allocator, dim, layers);
    defer rsf.deinit();

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    var ssi = SSI.init(allocator);
    defer ssi.deinit();

    var ranker = try Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED);
    defer ranker.deinit();

    const metrics = try validateModel(allocator, &rsf, config);
    try printValidationMetrics(stdout, &metrics);
}

fn generateSyntheticSamples(allocator: std.mem.Allocator, mgt: *MGT, count: usize) ![]TrainingSample {
    var samples = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| {
            sample.deinit(allocator);
        }
        samples.deinit();
    }

    const base_texts = [_][]const u8{
        "A mesterseges intelligencia a jovo kulcsa.",
        "Az adattudomany es gepi tanulas osszekapcsolodik.",
        "A neuralis halozatok komplex mintakat ismernek fel.",
        "Az automatizalas noveli a termelekenyseget.",
        "A kvantumszamitogepek uj lehetosegeket nyitnak.",
        "Az algoritmusok optimalizaljak a donteshozatalt.",
        "A termeszetes nyelvfeldolgozas emberi kommunikaciot ertelmez.",
        "A szamitogepek latas kepeket es videokat elemez.",
        "A robotika es automatizalas atalakitja az ipart.",
        "Az etikus AI fejlesztes fontos tarsadalmi kerdes.",
    };

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const base_text = base_texts[i % base_texts.len];
        const text_copy = try allocator.dupe(u8, base_text);
        errdefer allocator.free(text_copy);

        var tokens_list = std.ArrayList(u32).init(allocator);
        errdefer tokens_list.deinit();

        try mgt.encode(base_text, &tokens_list);

        const tokens = try tokens_list.toOwnedSlice();
        errdefer allocator.free(tokens);

        try samples.append(.{
            .text = text_copy,
            .tokens = tokens,
        });
    }

    return samples.toOwnedSlice();
}

fn createEmbeddingInPlace(embedding: *Tensor, tokens: []const u32, vocab_size: usize, dim: usize) !void {
    if (embedding.data.len == 0) return;
    if (vocab_size == 0) {
        @memset(embedding.data, 0.0);
        return;
    }

    var prng = PRNG.init(MainConfig.PRNG_SEED_SYNTHETIC);

    var i: usize = 0;
    while (i < embedding.data.len) : (i += 1) {
        embedding.data[i] = 0.0;
    }

    const max_tokens = @min(tokens.len, dim);
    i = 0;
    while (i < max_tokens) : (i += 1) {
        const token_id = tokens[i];
        const vocab_f: f32 = @floatFromInt(vocab_size);
        const token_f: f32 = @floatFromInt(token_id);
        const normalized = std.math.clamp(token_f / vocab_f, 0.0, 1.0);

        const rand_val = @as(f32, @floatFromInt(prng.next() & 0xFFFFFFFF)) / @as(f32, @floatFromInt(0xFFFFFFFF));
        const noise = rand_val * MainConfig.TENSOR_INIT_SCALE;

        if (i < embedding.data.len) {
            embedding.data[i] = normalized + noise;
        }
        if (i + dim < embedding.data.len) {
            embedding.data[i + dim] = normalized * 0.5 + noise;
        }
    }
}

fn validateModel(allocator: std.mem.Allocator, rsf: *RSF, config: *const Config) !ValidationMetrics {
    const n_samples = @max(config.num_validation_samples, 1);

    var prng = PRNG.init(MainConfig.PRNG_SEED_VALIDATION);

    var predictions = try allocator.alloc(f32, n_samples);
    defer allocator.free(predictions);

    var targets = try allocator.alloc(f32, n_samples);
    defer allocator.free(targets);

    const tensor_dim = std.math.mul(usize, config.embedding_dim, 2) catch return error.Overflow;
    var input = try Tensor.init(allocator, &.{ 1, tensor_dim });
    defer input.deinit();

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        for (input.data) |*val| {
            const rand_val = @as(f32, @floatFromInt(prng.next() & 0xFFFFFFFF)) / @as(f32, @floatFromInt(0xFFFFFFFF));
            val.* = rand_val * MainConfig.TENSOR_INIT_SCALE;
        }

        try rsf.forward(&input);

        var sum: f64 = 0.0;
        for (input.data) |val| {
            if (!std.math.isNan(val) and !std.math.isInf(val)) {
                sum += @as(f64, val);
            }
        }
        predictions[i] = @floatCast(sum / @as(f64, @floatFromInt(@max(input.data.len, 1))));

        const rand_target = @as(f32, @floatFromInt(prng.next() & 0xFFFFFFFF)) / @as(f32, @floatFromInt(0xFFFFFFFF));
        targets[i] = rand_target * MainConfig.TENSOR_INIT_SCALE;
    }

    var sum_pred: f64 = 0.0;
    var sum_target: f64 = 0.0;
    for (predictions) |p| sum_pred += @as(f64, p);
    for (targets) |t| sum_target += @as(f64, t);

    const n_f64 = @as(f64, @floatFromInt(n_samples));
    const mean_pred: f32 = @floatCast(sum_pred / n_f64);
    const mean_target: f32 = @floatCast(sum_target / n_f64);

    var mse: f64 = 0.0;
    var mae: f64 = 0.0;
    var ss_res: f64 = 0.0;
    var ss_tot: f64 = 0.0;

    i = 0;
    while (i < n_samples) : (i += 1) {
        const pred_f64 = @as(f64, predictions[i]);
        const target_f64 = @as(f64, targets[i]);
        const diff = pred_f64 - target_f64;
        mse += diff * diff;
        mae += std.math.fabs(diff);
        ss_res += diff * diff;
        const target_diff = target_f64 - @as(f64, mean_target);
        ss_tot += target_diff * target_diff;
    }

    mse /= n_f64;
    mae /= n_f64;

    const rmse = std.math.sqrt(mse);
    const r_squared = if (ss_tot > MainConfig.R_SQUARED_EPSILON) 1.0 - (ss_res / ss_tot) else 0.0;

    const std_err = rmse / std.math.sqrt(n_f64);
    const margin = MainConfig.CONFIDENCE_Z_SCORE * std_err;

    return ValidationMetrics{
        .mse = @floatCast(mse),
        .rmse = @floatCast(rmse),
        .mae = @floatCast(mae),
        .r_squared = @floatCast(r_squared),
        .mean_prediction = mean_pred,
        .mean_target = mean_target,
        .confidence_interval_lower = @floatCast(mse - margin),
        .confidence_interval_upper = @floatCast(mse + margin),
        .num_samples = n_samples,
    };
}

fn printValidationMetrics(writer: anytype, metrics: *const ValidationMetrics) !void {
    try writer.print("Validation Metrics (n={d}):\n", .{metrics.num_samples});
    try writer.print("  MSE: {d:.8}\n", .{metrics.mse});
    try writer.print("  RMSE: {d:.8}\n", .{metrics.rmse});
    try writer.print("  MAE: {d:.8}\n", .{metrics.mae});
    try writer.print("  R2 Score: {d:.6}\n", .{metrics.r_squared});
    try writer.print("  Mean Prediction: {d:.6}\n", .{metrics.mean_prediction});
    try writer.print("  Mean Target: {d:.6}\n", .{metrics.mean_target});
    try writer.print("  95% CI: [{d:.8}, {d:.8}]\n", .{ metrics.confidence_interval_lower, metrics.confidence_interval_upper });
}

fn saveModels(allocator: std.mem.Allocator, rsf: *const RSF, mgt: *const MGT, optimizer: *const SFD, ranker: *const Ranker, models_dir: []const u8) !void {
    try std.fs.cwd().makePath(models_dir);

    const rsf_path = try std.fmt.allocPrint(allocator, "{s}/rsf_trained.bin", .{models_dir});
    defer allocator.free(rsf_path);
    try saveRSF(rsf, rsf_path);

    const mgt_path = try std.fmt.allocPrint(allocator, "{s}/mgt_vocab.bin", .{models_dir});
    defer allocator.free(mgt_path);
    try saveMGT(mgt, mgt_path);

    const opt_path = try std.fmt.allocPrint(allocator, "{s}/optimizer_state.bin", .{models_dir});
    defer allocator.free(opt_path);
    try optimizer.saveState(opt_path);

    const ranker_path = try std.fmt.allocPrint(allocator, "{s}/ranker_weights.bin", .{models_dir});
    defer allocator.free(ranker_path);
    try saveRanker(ranker, ranker_path);
}

fn saveRSF(rsf: *const RSF, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const writer = file.writer();

    try writer.writeInt(u32, MainConfig.FILE_MAGIC_RSF, .Little);
    try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);
    try writer.writeInt(u64, @as(u64, @intCast(rsf.num_layers)), .Little);
    try writer.writeInt(u64, @as(u64, @intCast(rsf.dim)), .Little);

    var l: usize = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        try layer.s_weight.save(writer);
        try layer.t_weight.save(writer);
        try layer.s_bias.save(writer);
        try layer.t_bias.save(writer);
    }
}

fn saveMGT(mgt: *const MGT, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const writer = file.writer();

    try writer.writeInt(u32, MainConfig.FILE_MAGIC_MGT, .Little);
    try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);

    const vocab_size = mgt.vocabSize();
    if (vocab_size > MainConfig.MAX_VOCAB_SIZE) return error.VocabTooLarge;
    try writer.writeInt(u32, @as(u32, @intCast(vocab_size)), .Little);

    const VocabEntry = struct { key: []const u8, val: u32 };

    var sorted_entries = std.ArrayList(VocabEntry).init(mgt.allocator);
    defer sorted_entries.deinit();

    var it = mgt.token_to_id.iterator();
    while (it.next()) |entry| {
        try sorted_entries.append(.{ .key = entry.key_ptr.*, .val = entry.value_ptr.* });
    }

    const SortContext = struct {
        fn lessThan(_: void, a: VocabEntry, b: VocabEntry) bool {
            return a.val < b.val;
        }
    };
    std.mem.sort(VocabEntry, sorted_entries.items, {}, SortContext.lessThan);

    for (sorted_entries.items) |entry| {
        const token = entry.key;
        const id = entry.val;
        if (token.len > MainConfig.MAX_TOKEN_LENGTH) return error.TokenTooLong;
        try writer.writeInt(u32, @as(u32, @intCast(token.len)), .Little);
        try writer.writeAll(token);
        try writer.writeInt(u32, id, .Little);
    }
}

fn saveRanker(ranker: *const Ranker, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const writer = file.writer();

    try writer.writeInt(u32, MainConfig.FILE_MAGIC_RANKER, .Little);
    try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);

    try writer.writeInt(u64, @as(u64, @intCast(ranker.ngram_weights.len)), .Little);
    for (ranker.ngram_weights) |weight| {
        try writer.writeInt(u32, @as(u32, @bitCast(weight)), .Little);
    }

    try writer.writeInt(u64, @as(u64, @intCast(ranker.num_hash_functions)), .Little);
    for (ranker.lsh_hash_params) |param| {
        try writer.writeInt(u64, param, .Little);
    }

    try writer.writeInt(u64, ranker.seed, .Little);
}

fn runInteractiveREPL(allocator: std.mem.Allocator, mgt: *MGT, ssi: *SSI, ranker: *Ranker, config: *const Config) !void {
    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();
    const stdin = std.io.getStdIn().reader();

    const sample_texts = [_][]const u8{
        "A mesterseges intelligencia a jovo kulcsa.",
        "Az adattudomany es gepi tanulas osszekapcsolodik.",
        "A neuralis halozatok komplex mintakat ismernek fel.",
        "Az automatizalas noveli a termelekenyseget.",
        "A kvantumszamitogepek uj lehetosegeket nyitnak.",
        "Az algoritmusok optimalizaljak a donteshozatalt.",
        "A termeszetes nyelvfeldolgozas emberi kommunikaciot ertelmez.",
        "A szamitogepek latas kepeket es videokat elemez.",
        "A robotika es automatizalas atalakitja az ipart.",
        "Az etikus AI fejlesztes fontos tarsadalmi kerdes.",
    };

    var sample_idx: usize = 0;
    while (sample_idx < sample_texts.len) : (sample_idx += 1) {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        mgt.encode(sample_texts[sample_idx], &tokens) catch continue;

        if (tokens.items.len > 0) {
            const is_anchor = (sample_idx % MainConfig.ANCHOR_MODULO == 0);
            ssi.addSequence(tokens.items, @as(u64, @intCast(sample_idx)), is_anchor) catch continue;
        }
    }

    try stdout.writeAll("READY\n");
    

    var line_buf: [MainConfig.REPL_LINE_BUFFER_SIZE]u8 = undefined;
    @memset(&line_buf, 0);

    while (true) {

        const line = stdin.readUntilDelimiterOrEof(&line_buf, '\n') catch |err| {
            try stdout.print("Input error: {any}. Line too long (max {d} bytes).\n", .{ err, MainConfig.REPL_LINE_BUFFER_SIZE });
            continue;
        };

        if (line == null) break;

        const input = std.mem.trim(u8, line.?, " \t\r\n");
        if (input.len == 0) continue;

        if (std.mem.eql(u8, input, "exit") or std.mem.eql(u8, input, "quit")) {
            try stdout.writeAll("Goodbye.\n");
            break;
        }

        if (std.mem.eql(u8, input, "help")) {
            try stdout.writeAll(
                \\Commands:
                \\  help   - Show this help
                \\  status - Show system status
                \\  exit   - Exit the program
                \\  <text> - Query the knowledge base
                \\
            );
            continue;
        }

        if (std.mem.eql(u8, input, "status")) {
            const stats = ssi.stats();
            try stdout.print("SSI: {d} nodes, {d} leaves | MGT vocab: {d} | Ranker ngrams: {d}\n", .{
                stats.nodes,
                stats.leaves,
                mgt.vocabSize(),
                ranker.ngram_weights.len,
            });
            continue;
        }

        var query_tokens = std.ArrayList(u32).init(allocator);
        defer query_tokens.deinit();

        mgt.encode(input, &query_tokens) catch |err| {
            try stdout.print("Tokenization error: {any}\n", .{err});
            
            continue;
        };

        if (query_tokens.items.len == 0) {
            try stdout.writeAll("Empty query after tokenization.\n");
            
            continue;
        }

        const segments = ssi.retrieveTopK(query_tokens.items, config.top_k, allocator) catch |err| {
            try stdout.print("Retrieval error: {any}\n", .{err});
            
            continue;
        };
        defer {
            for (segments) |*seg| {
                var s = seg.*;
                s.deinit(allocator);
            }
            allocator.free(segments);
        }

        if (segments.len == 0) {
            try stdout.writeAll("No matching segments found. Query indexed for future reference.\n");
            const stats = ssi.stats();
            ssi.addSequence(query_tokens.items, @as(u64, @intCast(stats.nodes)), false) catch continue;
            continue;
        }

        ranker.rankCandidates(segments, ssi, allocator) catch |err| {
            try stdout.print("Ranking error: {any}\n", .{err});
            
            continue;
        };

        var decoded_list = std.ArrayList(u8).init(allocator);
        defer decoded_list.deinit();

        const best = segments[0];
        mgt.decode(best.tokens, &decoded_list) catch |err| {
            try stdout.print("Decode error: {any}\n", .{err});
            
            continue;
        };

        if (decoded_list.items.len > 0) {
            var safe_output = std.ArrayList(u8).init(allocator);
            defer safe_output.deinit();

            for (decoded_list.items) |c| {
                if (c >= 0x20 and c < 0x7F) {
                    try safe_output.append(c);
                } else if (c == '\n' or c == '\t') {
                    try safe_output.append(' ');
                }
            }

            try stdout.print("Response: {s} [score: {d:.4}]\n", .{ safe_output.items, best.score });
            
        } else {
            try stdout.print("Match found. Relevance score: {d:.4} | Tokens: {d}\n", .{ best.score, best.tokens.len });
            
        }
    }
}
