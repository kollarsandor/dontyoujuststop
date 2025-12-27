const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const Allocator = mem.Allocator;
const RSF = @import("../processor/rsf.zig").RSF;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const Tensor = @import("tensor.zig").Tensor;

pub const ModelError = error{
    InvalidMagicHeader,
    UnsupportedVersion,
    CorruptedData,
    ChecksumMismatch,
    InvalidMetadata,
    MissingComponent,
};

pub const MAGIC_HEADER = "JAIDE40\x00";
pub const CURRENT_VERSION: u32 = 1;

pub const ModelMetadata = struct {
    model_name: []const u8,
    version: u32,
    created_timestamp: i64,
    rsf_layers: usize,
    rsf_dim: usize,
    ranker_ngrams: usize,
    ranker_lsh_tables: usize,
    mgt_vocab_size: usize,
    description: []const u8,

    pub fn toJson(self: *const ModelMetadata, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{");
        try writer.print("\"model_name\":\"{s}\",", .{self.model_name});
        try writer.print("\"version\":{d},", .{self.version});
        try writer.print("\"created_timestamp\":{d},", .{self.created_timestamp});
        try writer.print("\"rsf_layers\":{d},", .{self.rsf_layers});
        try writer.print("\"rsf_dim\":{d},", .{self.rsf_dim});
        try writer.print("\"ranker_ngrams\":{d},", .{self.ranker_ngrams});
        try writer.print("\"ranker_lsh_tables\":{d},", .{self.ranker_lsh_tables});
        try writer.print("\"mgt_vocab_size\":{d},", .{self.mgt_vocab_size});
        try writer.print("\"description\":\"{s}\"", .{self.description});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn fromJson(allocator: Allocator, json: []const u8) !ModelMetadata {
        var parser = std.json.Parser.init(allocator, false);
        defer parser.deinit();

        var tree = try parser.parse(json);
        defer tree.deinit();

        const root = tree.root;

        return ModelMetadata{
            .model_name = try allocator.dupe(u8, root.Object.get("model_name").?.String),
            .version = @intCast(root.Object.get("version").?.Integer),
            .created_timestamp = root.Object.get("created_timestamp").?.Integer,
            .rsf_layers = @intCast(root.Object.get("rsf_layers").?.Integer),
            .rsf_dim = @intCast(root.Object.get("rsf_dim").?.Integer),
            .ranker_ngrams = @intCast(root.Object.get("ranker_ngrams").?.Integer),
            .ranker_lsh_tables = @intCast(root.Object.get("ranker_lsh_tables").?.Integer),
            .mgt_vocab_size = @intCast(root.Object.get("mgt_vocab_size").?.Integer),
            .description = try allocator.dupe(u8, root.Object.get("description").?.String),
        };
    }

    pub fn deinit(self: *ModelMetadata, allocator: Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.description);
    }
};

pub const ModelFormat = struct {
    metadata: ModelMetadata,
    rsf: ?*RSF = null,
    ranker: ?*Ranker = null,
    mgt: ?*MGT = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, description: []const u8) !ModelFormat {
        const timestamp = std.time.timestamp();

        return ModelFormat{
            .metadata = ModelMetadata{
                .model_name = try allocator.dupe(u8, name),
                .version = CURRENT_VERSION,
                .created_timestamp = timestamp,
                .rsf_layers = 0,
                .rsf_dim = 0,
                .ranker_ngrams = 0,
                .ranker_lsh_tables = 0,
                .mgt_vocab_size = 0,
                .description = try allocator.dupe(u8, description),
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ModelFormat) void {
        self.metadata.deinit(self.allocator);
    }

    pub fn setRSF(self: *ModelFormat, rsf: *RSF) void {
        self.rsf = rsf;
        self.metadata.rsf_layers = rsf.num_layers;
        self.metadata.rsf_dim = rsf.dim;
    }

    pub fn setRanker(self: *ModelFormat, ranker: *Ranker) void {
        self.ranker = ranker;
        self.metadata.ranker_ngrams = ranker.num_ngrams;
        self.metadata.ranker_lsh_tables = ranker.num_hash_functions;
    }

    pub fn setMGT(self: *ModelFormat, mgt: *MGT) void {
        self.mgt = mgt;
        self.metadata.mgt_vocab_size = mgt.vocabSize();
    }
};

pub fn exportModel(model: *ModelFormat, path: []const u8) !void {
    var file = try fs.cwd().createFile(path, .{});
    defer file.close();

    var buffered_writer = std.io.bufferedWriter(file.writer());
    const writer = buffered_writer.writer();

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var counting_writer = std.io.countingWriter(writer);
    const cw = counting_writer.writer();

    try cw.writeAll(MAGIC_HEADER);
    hasher.update(MAGIC_HEADER);

    try cw.writeInt(u32, CURRENT_VERSION, .Little);
    hasher.update(&std.mem.toBytes(CURRENT_VERSION));

    const metadata_json = try model.metadata.toJson(model.allocator);
    defer model.allocator.free(metadata_json);

    try cw.writeInt(u32, @intCast(metadata_json.len), .Little);
    hasher.update(&std.mem.toBytes(@as(u32, @intCast(metadata_json.len))));

    try cw.writeAll(metadata_json);
    hasher.update(metadata_json);

    if (model.rsf) |rsf| {
        try cw.writeInt(u8, 1, .Little);
        hasher.update(&[_]u8{1});

        var rsf_buf = std.ArrayList(u8).init(model.allocator);
        defer rsf_buf.deinit();
        var rsf_writer = rsf_buf.writer();

        try rsf_writer.writeInt(usize, rsf.num_layers, .Little);
        try rsf_writer.writeInt(usize, rsf.dim, .Little);

        for (rsf.layers) |layer| {
            try layer.s_weight.save(rsf_writer);
            try layer.t_weight.save(rsf_writer);
            try layer.s_bias.save(rsf_writer);
            try layer.t_bias.save(rsf_writer);
        }

        const rsf_data = try rsf_buf.toOwnedSlice();
        defer model.allocator.free(rsf_data);

        try cw.writeInt(u32, @intCast(rsf_data.len), .Little);
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(rsf_data.len))));
        try cw.writeAll(rsf_data);
        hasher.update(rsf_data);
    } else {
        try cw.writeInt(u8, 0, .Little);
        hasher.update(&[_]u8{0});
    }

    if (model.ranker) |ranker| {
        try cw.writeInt(u8, 1, .Little);
        hasher.update(&[_]u8{1});

        var ranker_buf = std.ArrayList(u8).init(model.allocator);
        defer ranker_buf.deinit();
        var ranker_writer = ranker_buf.writer();

        try ranker_writer.writeInt(u8, 1, .Little);
        try ranker_writer.writeInt(usize, ranker.ngram_weights.len, .Little);

        for (ranker.ngram_weights) |w| {
            try ranker_writer.writeAll(mem.asBytes(&w));
        }

        try ranker_writer.writeInt(usize, ranker.num_hash_functions, .Little);
        for (ranker.lsh_hash_params) |t| {
            try ranker_writer.writeInt(u64, t, .Little);
        }

        try ranker_writer.writeInt(u64, ranker.seed, .Little);

        const ranker_data = try ranker_buf.toOwnedSlice();
        defer model.allocator.free(ranker_data);

        try cw.writeInt(u32, @intCast(ranker_data.len), .Little);
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(ranker_data.len))));
        try cw.writeAll(ranker_data);
        hasher.update(ranker_data);
    } else {
        try cw.writeInt(u8, 0, .Little);
        hasher.update(&[_]u8{0});
    }

    if (model.mgt) |mgt| {
        try cw.writeInt(u8, 1, .Little);
        hasher.update(&[_]u8{1});

        var mgt_buf = std.ArrayList(u8).init(model.allocator);
        defer mgt_buf.deinit();
        var mgt_writer = mgt_buf.writer();

        const vocab_size = mgt.vocabSize();
        try mgt_writer.writeInt(u32, @intCast(vocab_size), .Little);

        var it = mgt.token_to_id.iterator();
        while (it.next()) |entry| {
            const word = entry.key_ptr.*;
            try mgt_writer.writeInt(u32, @intCast(word.len), .Little);
            try mgt_writer.writeAll(word);
        }

        const mgt_data = try mgt_buf.toOwnedSlice();
        defer model.allocator.free(mgt_data);

        try cw.writeInt(u32, @intCast(mgt_data.len), .Little);
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(mgt_data.len))));
        try cw.writeAll(mgt_data);
        hasher.update(mgt_data);
    } else {
        try cw.writeInt(u8, 0, .Little);
        hasher.update(&[_]u8{0});
    }

    var checksum: [32]u8 = undefined;
    hasher.final(&checksum);
    try writer.writeAll(&checksum);

    try buffered_writer.flush();
}

pub fn importModel(path: []const u8, allocator: Allocator) !ModelFormat {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});

    var magic: [8]u8 = undefined;
    try reader.readNoEof(&magic);
    if (!mem.eql(u8, &magic, MAGIC_HEADER)) {
        return ModelError.InvalidMagicHeader;
    }
    hasher.update(&magic);

    const version = try reader.readInt(u32, .Little);
    hasher.update(&std.mem.toBytes(version));
    if (version != CURRENT_VERSION) {
        return ModelError.UnsupportedVersion;
    }

    const metadata_len = try reader.readInt(u32, .Little);
    hasher.update(&std.mem.toBytes(metadata_len));

    const metadata_json = try allocator.alloc(u8, metadata_len);
    defer allocator.free(metadata_json);
    try reader.readNoEof(metadata_json);
    hasher.update(metadata_json);

    var metadata = try ModelMetadata.fromJson(allocator, metadata_json);

    var model = ModelFormat{
        .metadata = metadata,
        .allocator = allocator,
    };

    const has_rsf = try reader.readInt(u8, .Little);
    hasher.update(&[_]u8{has_rsf});

    if (has_rsf == 1) {
        const rsf_len = try reader.readInt(u32, .Little);
        hasher.update(&std.mem.toBytes(rsf_len));

        const rsf_data = try allocator.alloc(u8, rsf_len);
        defer allocator.free(rsf_data);
        try reader.readNoEof(rsf_data);
        hasher.update(rsf_data);

        var rsf_stream = std.io.fixedBufferStream(rsf_data);
        var rsf_reader = rsf_stream.reader();

        const num_layers = try rsf_reader.readInt(usize, .Little);
        const dim = try rsf_reader.readInt(usize, .Little);

        var rsf = try allocator.create(RSF);
        rsf.* = try RSF.init(allocator, dim, num_layers);

        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            rsf.layers[l].s_weight.deinit();
            rsf.layers[l].t_weight.deinit();
            rsf.layers[l].s_bias.deinit();
            rsf.layers[l].t_bias.deinit();

            rsf.layers[l].s_weight = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].t_weight = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].s_bias = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].t_bias = try Tensor.load(allocator, rsf_reader);
        }

        model.rsf = rsf;
    }

    const has_ranker = try reader.readInt(u8, .Little);
    hasher.update(&[_]u8{has_ranker});

    if (has_ranker == 1) {
        const ranker_len = try reader.readInt(u32, .Little);
        hasher.update(&std.mem.toBytes(ranker_len));

        const ranker_data = try allocator.alloc(u8, ranker_len);
        defer allocator.free(ranker_data);
        try reader.readNoEof(ranker_data);
        hasher.update(ranker_data);

        var ranker_stream = std.io.fixedBufferStream(ranker_data);
        var ranker_reader = ranker_stream.reader();

        const ranker_version = try ranker_reader.readInt(u8, .Little);
        if (ranker_version != 1) return ModelError.UnsupportedVersion;

        const num_weights = try ranker_reader.readInt(usize, .Little);

        var ranker = try allocator.create(Ranker);
        const ngram_weights_alloc = try allocator.alloc(f32, num_weights);
        @memset(ngram_weights_alloc, 0);
        ranker.* = Ranker{
            .ngram_weights = ngram_weights_alloc,
            .lsh_hash_params = &[_]u64{},
            .num_hash_functions = 0,
            .num_ngrams = num_weights,
            .seed = 0,
            .allocator = allocator,
        };

        for (ranker.ngram_weights) |*w| {
            var bytes: [@sizeOf(f32)]u8 = undefined;
            try ranker_reader.readNoEof(&bytes);
            w.* = @as(*const f32, @ptrCast(&bytes)).*;
        }

        const num_hash_funcs = try ranker_reader.readInt(usize, .Little);
        ranker.num_hash_functions = num_hash_funcs;
        ranker.lsh_hash_params = try allocator.alloc(u64, num_hash_funcs * 2);

        for (ranker.lsh_hash_params) |*t| {
            t.* = try ranker_reader.readInt(u64, .Little);
        }

        ranker.seed = try ranker_reader.readInt(u64, .Little);

        model.ranker = ranker;
    }

    const has_mgt = try reader.readInt(u8, .Little);
    hasher.update(&[_]u8{has_mgt});

    if (has_mgt == 1) {
        const mgt_len = try reader.readInt(u32, .Little);
        hasher.update(&std.mem.toBytes(mgt_len));

        const mgt_data = try allocator.alloc(u8, mgt_len);
        defer allocator.free(mgt_data);
        try reader.readNoEof(mgt_data);
        hasher.update(mgt_data);

        var mgt_stream = std.io.fixedBufferStream(mgt_data);
        var mgt_reader = mgt_stream.reader();

        const vocab_size = try mgt_reader.readInt(u32, .Little);

        var words_list = std.ArrayList([]u8).init(allocator);
        defer {
            for (words_list.items) |w| allocator.free(w);
            words_list.deinit();
        }

        var i: usize = 0;
        while (i < vocab_size) : (i += 1) {
            const word_len = try mgt_reader.readInt(u32, .Little);
            var word = try allocator.alloc(u8, word_len);
            try mgt_reader.readNoEof(word);
            try words_list.append(word);
        }

        const words_const = try allocator.alloc([]const u8, words_list.items.len);
        defer allocator.free(words_const);
        var idx: usize = 0;
        while (idx < words_list.items.len) : (idx += 1) {
            words_const[idx] = words_list.items[idx];
        }

        var mgt = try allocator.create(MGT);
        mgt.* = try MGT.init(allocator, words_const, &.{});

        model.mgt = mgt;
    }

    var expected_checksum: [32]u8 = undefined;
    hasher.final(&expected_checksum);

    var stored_checksum: [32]u8 = undefined;
    try reader.readNoEof(&stored_checksum);

    if (!mem.eql(u8, &expected_checksum, &stored_checksum)) {
        model.deinit();
        if (model.rsf) |rsf| {
            rsf.deinit();
            allocator.destroy(rsf);
        }
        if (model.ranker) |ranker| {
            ranker.deinit();
            allocator.destroy(ranker);
        }
        if (model.mgt) |mgt| {
            mgt.deinit();
            allocator.destroy(mgt);
        }
        return ModelError.ChecksumMismatch;
    }

    return model;
}

pub fn freeLoadedModel(model: *ModelFormat) void {
    if (model.rsf) |rsf| {
        rsf.deinit();
        model.allocator.destroy(rsf);
    }
    if (model.ranker) |ranker| {
        ranker.deinit();
        model.allocator.destroy(ranker);
    }
    if (model.mgt) |mgt| {
        mgt.deinit();
        model.allocator.destroy(mgt);
    }
    model.deinit();
}

test "ModelFormat creation and metadata" {
    const testing = std.testing;
    var gpa = testing.allocator;

    var model = try ModelFormat.init(gpa, "TestModel", "A test model");
    defer model.deinit();

    try testing.expectEqualStrings("TestModel", model.metadata.model_name);
    try testing.expectEqualStrings("A test model", model.metadata.description);
    try testing.expectEqual(CURRENT_VERSION, model.metadata.version);
}

test "Metadata JSON serialization" {
    const testing = std.testing;
    var gpa = testing.allocator;

    var metadata = ModelMetadata{
        .model_name = try gpa.dupe(u8, "Test"),
        .version = 1,
        .created_timestamp = 1234567890,
        .rsf_layers = 4,
        .rsf_dim = 128,
        .ranker_ngrams = 5,
        .ranker_lsh_tables = 8,
        .mgt_vocab_size = 1000,
        .description = try gpa.dupe(u8, "Test model"),
    };
    defer metadata.deinit(gpa);

    const json = try metadata.toJson(gpa);
    defer gpa.free(json);

    try testing.expect(json.len > 0);
}
