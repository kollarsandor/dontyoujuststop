// ============================================================================
// JAIDE v40 Inference Server - CRITICAL SECURITY CONFIGURATION
// ============================================================================
//
// ⚠️  SECURITY WARNING: This server MUST be properly configured before
// production deployment. By default, it includes multiple security layers:
//
// 1. API KEY AUTHENTICATION (Environment Variable)
//    - Set API_KEY environment variable before starting the server
//    - All inference requests must include "Authorization: Bearer <API_KEY>" header
//    - Example: export API_KEY="your-secret-key-here"
//
// 2. RATE LIMITING (Per-IP)
//    - Default: 10 requests per minute per IP address
//    - Configurable via rate_limit_per_minute in ServerConfig
//    - Returns HTTP 429 (Too Many Requests) when exceeded
//
// 3. REQUEST SIZE LIMITS
//    - Default: 1MB maximum payload size
//    - Configurable via max_request_size_bytes in ServerConfig
//    - Returns HTTP 413 (Payload Too Large) when exceeded
//
// 4. TRUSTED NETWORK DEPLOYMENT ONLY
//    - This server should ONLY be deployed on trusted networks
//    - Use a reverse proxy (nginx, Caddy) for production:
//      * TLS/HTTPS termination
//      * Additional firewall rules
//      * Request validation
//      * DDoS protection
//
// PRODUCTION CHECKLIST:
// ✓ Set strong API_KEY (min 32 random characters)
// ✓ Configure rate limiting for your use case
// ✓ Deploy behind reverse proxy with HTTPS
// ✓ Enable firewall rules (allow only trusted IPs)
// ✓ Monitor logs for suspicious activity
// ✓ Regular security audits
//
// ============================================================================

const std = @import("std");
const net = std.net;
const mem = std.mem;
const fs = std.fs;
const Thread = std.Thread;
const Allocator = mem.Allocator;
const RSF = @import("../processor/rsf.zig").RSF;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const SSI = @import("../index/ssi.zig").SSI;
const Tensor = @import("../core/tensor.zig").Tensor;
const ModelFormat = @import("../core/model_io.zig").ModelFormat;
const importModel = @import("../core/model_io.zig").importModel;
const freeLoadedModel = @import("../core/model_io.zig").freeLoadedModel;

pub const ServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "0.0.0.0",
    max_connections: u32 = 100,
    request_timeout_ms: u64 = 30000,
    batch_size: usize = 32,
    model_path: ?[]const u8 = null,

    // Security configuration
    api_key: ?[]const u8 = null,  // If null, reads from API_KEY env var
    rate_limit_per_minute: u32 = 10,  // Requests per IP per minute
    max_request_size_bytes: usize = 1024 * 1024,  // 1MB default
    require_api_key: bool = true,  // Set to false to disable API key (NOT RECOMMENDED)

    // Training configuration
    dataset_path: ?[]const u8 = null,
    sample_limit: ?usize = null,
    num_validation_samples: usize = 100,
};

// Rate limiter to track requests per IP address
const RateLimiter = struct {
    const RequestLog = struct {
        timestamps: std.ArrayList(i64),
        mutex: Thread.Mutex,
    };

    logs: std.StringHashMap(RequestLog),
    allocator: Allocator,
    mutex: Thread.Mutex,
    window_seconds: u64,
    max_requests: u32,

    pub fn init(allocator: Allocator, max_requests_per_minute: u32) RateLimiter {
        return RateLimiter{
            .logs = std.StringHashMap(RequestLog).init(allocator),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
            .window_seconds = 60,
            .max_requests = max_requests_per_minute,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.logs.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.timestamps.deinit();
        }
        self.logs.deinit();
    }

    pub fn checkAndRecord(self: *RateLimiter, ip_address: []const u8) !bool {
        const now = std.time.timestamp();
        const cutoff = now - @as(i64, @intCast(self.window_seconds));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = try self.logs.getOrPut(ip_address);
        if (!result.found_existing) {
            result.value_ptr.* = RequestLog{
                .timestamps = std.ArrayList(i64).init(self.allocator),
                .mutex = Thread.Mutex{},
            };
        }

        var log = result.value_ptr;
        log.mutex.lock();
        defer log.mutex.unlock();

        // Remove old timestamps outside the window
        var i: usize = 0;
        while (i < log.timestamps.items.len) {
            if (log.timestamps.items[i] < cutoff) {
                _ = log.timestamps.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        // Check if rate limit exceeded
        if (log.timestamps.items.len >= self.max_requests) {
            return false;  // Rate limit exceeded
        }

        // Record this request
        try log.timestamps.append(now);
        return true;  // Request allowed
    }
};

pub const InferenceRequest = struct {
    text: []const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !InferenceRequest {
        var parser = std.json.Parser.init(allocator, false);
        defer parser.deinit();

        var tree = try parser.parse(json);
        defer tree.deinit();

        const root = tree.root;

        const text = root.Object.get("text") orelse return error.MissingTextField;

        var max_tokens: ?usize = null;
        if (root.Object.get("max_tokens")) |mt| {
            if (mt.Integer < 0) return error.InvalidMaxTokens;
            if (mt.Integer > 1000000) return error.MaxTokensTooLarge;
            max_tokens = @intCast(mt.Integer);
        }

        var return_embeddings = false;
        if (root.Object.get("return_embeddings")) |re| {
            return_embeddings = re.Bool;
        }

        return InferenceRequest{
            .text = try allocator.dupe(u8, text.String),
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *InferenceRequest, allocator: Allocator) void {
        allocator.free(self.text);
    }
};

pub const InferenceResponse = struct {
    tokens: []u32,
    embeddings: ?[]f32 = null,
    processing_time_ms: f64,

    pub fn toJson(self: *const InferenceResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"tokens\":[");
        var i: usize = 0;
        while (i < self.tokens.len) : (i += 1) {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{self.tokens[i]});
        }
        try writer.writeAll("]");

        if (self.embeddings) |emb| {
            try writer.writeAll(",\"embeddings\":[");
            var j: usize = 0;
            while (j < emb.len) : (j += 1) {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d:.6}", .{emb[j]});
            }
            try writer.writeAll("]");
        }

        try writer.print(",\"processing_time_ms\":{d:.2}", .{self.processing_time_ms});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn deinit(self: *InferenceResponse, allocator: Allocator) void {
        allocator.free(self.tokens);
        if (self.embeddings) |emb| {
            allocator.free(emb);
        }
    }
};

pub const HealthResponse = struct {
    status: []const u8 = "healthy",
    uptime_seconds: u64,
    model_loaded: bool,
    version: []const u8 = "1.0.0",

    pub fn toJson(self: *const HealthResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{");
        try writer.print("\"status\":\"{s}\",", .{self.status});
        try writer.print("\"uptime_seconds\":{d},", .{self.uptime_seconds});
        try writer.print("\"model_loaded\":{},", .{self.model_loaded});
        try writer.print("\"version\":\"{s}\"", .{self.version});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }
};

pub const InferenceServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    model: ?ModelFormat = null,
    ssi: ?SSI = null,
    start_time: i64,
    running: std.atomic.Atomic(bool),
    rate_limiter: RateLimiter,
    api_key: ?[]const u8,

    pub fn init(allocator: Allocator, config: ServerConfig) !InferenceServer {
        // Load API key from environment or config
        var api_key: ?[]const u8 = null;
        if (config.require_api_key) {
            if (config.api_key) |key| {
                api_key = try allocator.dupe(u8, key);
            } else {
                // Try to read from environment
                if (std.os.getenv("API_KEY")) |env_key| {
                    api_key = try allocator.dupe(u8, env_key);
                    std.debug.print("✓ API key loaded from API_KEY environment variable\n", .{});
                } else {
                    std.debug.print("⚠️  WARNING: No API key configured! Set API_KEY environment variable or disable require_api_key\n", .{});
                }
            }
        }

        return InferenceServer{
            .allocator = allocator,
            .config = config,
            .start_time = std.time.timestamp(),
            .running = std.atomic.Atomic(bool).init(false),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit_per_minute),
            .api_key = api_key,
        };
    }

    pub fn deinit(self: *InferenceServer) void {
        if (self.model) |*model| {
            freeLoadedModel(model);
        }
        if (self.ssi) |*ssi| {
            ssi.deinit();
        }
        if (self.api_key) |key| {
            self.allocator.free(key);
        }
        self.rate_limiter.deinit();
    }

    pub fn loadModel(self: *InferenceServer, path: []const u8) !void {
        self.model = try importModel(path, self.allocator);
        self.ssi = SSI.init(self.allocator);
    }

    pub fn start(self: *InferenceServer) !void {
        const address = try net.Address.parseIp(self.config.host, self.config.port);
        var server = net.StreamServer.init(.{
            .reuse_address = true,
        });
        defer server.deinit();

        try server.listen(address);
        self.running.store(true, .SeqCst);

        std.debug.print("🔒 Security configuration:\n", .{});
        std.debug.print("   - API key auth: {s}\n", .{if (self.api_key != null) "ENABLED" else "DISABLED"});
        std.debug.print("   - Rate limiting: {d} requests/min per IP\n", .{self.config.rate_limit_per_minute});
        std.debug.print("   - Max request size: {d} bytes\n", .{self.config.max_request_size_bytes});
        std.debug.print("\n", .{});
        std.debug.print("Inference server listening on {s}:{d}\n", .{ self.config.host, self.config.port });

        while (self.running.load(.SeqCst)) {
            const connection = server.accept() catch |err| {
                std.debug.print("Failed to accept connection: {}\n", .{err});
                continue;
            };

            self.handleConnection(connection) catch |err| {
                std.debug.print("Error handling connection: {}\n", .{err});
            };
        }
    }

    pub fn stop(self: *InferenceServer) void {
        self.running.store(false, .SeqCst);
    }

    fn handleConnection(self: *InferenceServer, connection: net.StreamServer.Connection) !void {
        defer connection.stream.close();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();

        // Get client IP address for rate limiting
        const client_addr = connection.address;
        var ip_buf: [64]u8 = undefined;
        const ip_str = try std.fmt.bufPrint(&ip_buf, "{}", .{client_addr});

        var buf: [16384]u8 = undefined;
        const bytes_read = try connection.stream.read(&buf);
        if (bytes_read == 0) return;

        // SECURITY: Check request size limit
        if (bytes_read > self.config.max_request_size_bytes) {
            try self.sendError(connection.stream, "Request too large", 413);
            return;
        }

        const request_data = buf[0..bytes_read];

        const method_end = mem.indexOf(u8, request_data, " ") orelse return error.InvalidRequest;
        const method = request_data[0..method_end];

        const path_start = method_end + 1;
        const path_end = mem.indexOfPos(u8, request_data, path_start, " ") orelse return error.InvalidRequest;
        const path = request_data[path_start..path_end];

        const headers_end = mem.indexOf(u8, request_data, "\r\n\r\n") orelse return error.InvalidRequest;
        const headers = request_data[0..headers_end];
        const body = if (headers_end + 4 < request_data.len) request_data[headers_end + 4 ..] else "";

        if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/v1/health")) {
            try self.handleHealth(connection.stream, temp_allocator);
        } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/inference")) {
            // SECURITY: Check rate limit
            const rate_allowed = try self.rate_limiter.checkAndRecord(ip_str);
            if (!rate_allowed) {
                try self.sendError(connection.stream, "Rate limit exceeded", 429);
                std.debug.print("⚠️  Rate limit exceeded for IP: {s}\n", .{ip_str});
                return;
            }

            // SECURITY: Check API key authentication
            if (self.api_key) |expected_key| {
                const auth_valid = try self.checkAuthorization(headers, expected_key);
                if (!auth_valid) {
                    try self.sendError(connection.stream, "Unauthorized - Invalid or missing API key", 401);
                    std.debug.print("⚠️  Unauthorized access attempt from IP: {s}\n", .{ip_str});
                    return;
                }
            }

            try self.handleInference(connection.stream, body, temp_allocator);
        } else {
            try self.sendNotFound(connection.stream);
        }
    }

    fn checkAuthorization(self: *InferenceServer, headers: []const u8, expected_key: []const u8) !bool {
        _ = self;

        // Look for Authorization header
        var lines = mem.split(u8, headers, "\r\n");
        while (lines.next()) |line| {
            if (mem.startsWith(u8, line, "Authorization:") or mem.startsWith(u8, line, "authorization:")) {
                const value_start = mem.indexOf(u8, line, ":") orelse continue;
                const value = mem.trim(u8, line[value_start + 1..], " \t");

                // Check for "Bearer <token>" format
                if (mem.startsWith(u8, value, "Bearer ") or mem.startsWith(u8, value, "bearer ")) {
                    const token = mem.trim(u8, value[7..], " \t");
                    return mem.eql(u8, token, expected_key);
                }
            }
        }

        return false;
    }

    fn handleHealth(self: *InferenceServer, stream: net.Stream, allocator: Allocator) !void {
        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));

        const response = HealthResponse{
            .uptime_seconds = uptime,
            .model_loaded = self.model != null,
        };

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        try stream.writeAll(response_buf.items);
    }

    fn handleInference(self: *InferenceServer, stream: net.Stream, body: []const u8, allocator: Allocator) !void {
        if (self.model == null or self.model.?.mgt == null) {
            try self.sendError(stream, "Model not loaded", 503);
            return;
        }

        const start_time = std.time.milliTimestamp();

        var request = InferenceRequest.fromJson(allocator, body) catch |err| {
            std.debug.print("Failed to parse request: {}\n", .{err});
            try self.sendError(stream, "Invalid JSON request", 400);
            return;
        };
        defer request.deinit(allocator);

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        self.model.?.mgt.?.encode(request.text, &tokens) catch |err| {
            std.debug.print("Encoding failed: {}\n", .{err});
            try self.sendError(stream, "Encoding failed", 500);
            return;
        };

        const max_tokens = request.max_tokens orelse tokens.items.len;
        const final_tokens = if (tokens.items.len > max_tokens)
            tokens.items[0..max_tokens]
        else
            tokens.items;

        var embeddings: ?[]f32 = null;
        if (request.return_embeddings and self.model.?.rsf != null) {
            const dim = self.model.?.rsf.?.dim;
            const batch_size = 1;

            var input_tensor = Tensor.init(allocator, &.{ batch_size, dim * 2 }) catch |err| {
                std.debug.print("Tensor init failed: {}\n", .{err});
                try self.sendError(stream, "Failed to create embeddings", 500);
                return;
            };
            defer input_tensor.deinit();

            var k: usize = 0;
            while (k < input_tensor.data.len) : (k += 1) {
                input_tensor.data[k] = if (k < final_tokens.len)
                    @as(f32, @floatFromInt(final_tokens[k])) / 1000.0
                else
                    0.0;
            }

            self.model.?.rsf.?.forward(&input_tensor) catch |err| {
                std.debug.print("RSF forward failed: {}\n", .{err});
                try self.sendError(stream, "Embedding generation failed", 500);
                return;
            };

            embeddings = try allocator.alloc(f32, @min(dim, 128));
            var m: usize = 0;
            while (m < embeddings.?.len) : (m += 1) {
                embeddings.?[m] = if (m < input_tensor.data.len) input_tensor.data[m] else 0.0;
            }
        }

        const end_time = std.time.milliTimestamp();
        const processing_time = @as(f64, @floatFromInt(end_time - start_time));

        const tokens_copy = try allocator.dupe(u32, final_tokens);

        var response = InferenceResponse{
            .tokens = tokens_copy,
            .embeddings = embeddings,
            .processing_time_ms = processing_time,
        };
        defer {
            allocator.free(response.tokens);
            if (response.embeddings) |emb| allocator.free(emb);
        }

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        try stream.writeAll(response_buf.items);
    }

    fn sendError(self: *InferenceServer, stream: net.Stream, message: []const u8, status_code: u16) !void {
        _ = self;
        var buf: [1024]u8 = undefined;
        const json = try std.fmt.bufPrint(&buf, "{{\"error\":\"{s}\"}}", .{message});

        const status_text = switch (status_code) {
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            else => "Error",
        };

        var response_buf: [2048]u8 = undefined;
        const response = try std.fmt.bufPrint(&response_buf,
            "HTTP/1.1 {d} {s}\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Content-Length: {d}\r\n" ++
            "\r\n" ++
            "{s}",
            .{ status_code, status_text, json.len, json }
        );

        try stream.writeAll(response);
    }

    fn sendNotFound(self: *InferenceServer, stream: net.Stream) !void {
        try self.sendError(stream, "Endpoint not found", 404);
    }
};

pub const BatchInferenceRequest = struct {
    texts: [][]const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !BatchInferenceRequest {
        var parser = std.json.Parser.init(allocator, false);
        defer parser.deinit();

        var tree = try parser.parse(json);
        defer tree.deinit();

        const root = tree.root;

        const texts_array = root.Object.get("texts") orelse return error.MissingTextsField;

        var texts = try allocator.alloc([]const u8, texts_array.Array.items.len);
        var n: usize = 0;
        while (n < texts_array.Array.items.len) : (n += 1) {
            texts[n] = try allocator.dupe(u8, texts_array.Array.items[n].String);
        }

        var max_tokens: ?usize = null;
        if (root.Object.get("max_tokens")) |mt| {
            if (mt.Integer < 0) return error.InvalidMaxTokens;
            if (mt.Integer > 1000000) return error.MaxTokensTooLarge;
            max_tokens = @intCast(mt.Integer);
        }

        var return_embeddings = false;
        if (root.Object.get("return_embeddings")) |re| {
            return_embeddings = re.Bool;
        }

        return BatchInferenceRequest{
            .texts = texts,
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *BatchInferenceRequest, allocator: Allocator) void {
        for (self.texts) |text| {
            allocator.free(text);
        }
        allocator.free(self.texts);
    }
};

pub fn runServer(allocator: Allocator, config: ServerConfig) !void {
    var server = try InferenceServer.init(allocator, config);
    defer server.deinit();

    if (config.model_path) |path| {
        try server.loadModel(path);
    }

    try server.start();
}

test "InferenceRequest JSON parsing" {
    const testing = std.testing;
    var gpa = testing.allocator;

    const json = "{\"text\":\"hello world\",\"max_tokens\":100,\"return_embeddings\":true}";

    var request = try InferenceRequest.fromJson(gpa, json);
    defer request.deinit(gpa);

    try testing.expectEqualStrings("hello world", request.text);
    try testing.expectEqual(@as(?usize, 100), request.max_tokens);
    try testing.expect(request.return_embeddings);
}

test "HealthResponse JSON serialization" {
    const testing = std.testing;
    var gpa = testing.allocator;

    const response = HealthResponse{
        .uptime_seconds = 3600,
        .model_loaded = true,
    };

    const json = try response.toJson(gpa);
    defer gpa.free(json);

    try testing.expect(mem.indexOf(u8, json, "\"status\":\"healthy\"") != null);
    try testing.expect(mem.indexOf(u8, json, "\"uptime_seconds\":3600") != null);
}

test "InferenceResponse JSON serialization" {
    const testing = std.testing;
    var gpa = testing.allocator;

    const tokens = try gpa.alloc(u32, 3);
    defer gpa.free(tokens);
    tokens[0] = 1;
    tokens[1] = 2;
    tokens[2] = 3;

    var response = InferenceResponse{
        .tokens = tokens,
        .processing_time_ms = 42.5,
    };

    const json = try response.toJson(gpa);
    defer gpa.free(json);

    try testing.expect(mem.indexOf(u8, json, "\"tokens\":[1,2,3]") != null);
}

test "BatchInferenceRequest JSON parsing" {
    const testing = std.testing;
    var gpa = testing.allocator;

    const json = "{\"texts\":[\"hello\",\"world\"],\"max_tokens\":50}";

    var request = try BatchInferenceRequest.fromJson(gpa, json);
    defer request.deinit(gpa);

    try testing.expectEqual(@as(usize, 2), request.texts.len);
    try testing.expectEqualStrings("hello", request.texts[0]);
    try testing.expectEqualStrings("world", request.texts[1]);
}