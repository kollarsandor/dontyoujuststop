const std = @import("std");
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const builtin = @import("builtin");
const Allocator = mem.Allocator;

pub const IoConfig = struct {
    pub const BUFFER_SIZE: usize = 8192;
    pub const LARGE_CHUNK_SIZE: usize = 65536;
    pub const MAX_READ_BYTES: usize = 100 * 1024 * 1024;
    pub const PAGE_SIZE: usize = mem.page_size;
    pub const WYHASH_PRIME_1: u64 = 0xff51afd7ed558ccd;
    pub const WYHASH_PRIME_2: u64 = 0xc4ceb9fe1a85ec53;
    pub const MIX_SHIFT: u6 = 33;
    pub const TRUNCATE_SHIFT: u6 = 32;
    pub const MAX_FLUSH_DEPTH: usize = 10;
    pub const DEFAULT_FILE_MODE: u9 = 0o644;
    pub const SECURE_FILE_MODE: u9 = 0o600;
};

pub const IoError = error{
    InvalidFileSize,
    FileTooLarge,
    FileIsEmpty,
    BufferNotMapped,
    OutOfBounds,
    RecursionDepthExceeded,
    MaxBytesExceeded,
    InvalidPathCharacter,
    EndOfStream,
    UnexpectedEndOfFile,
    FileNotFound,
    AccessDenied,
    PathAlreadyExists,
    InvalidPath,
    NotADirectory,
    NotAFile,
    OperationFailed,
};

fn generateRuntimeSeed() u64 {
    const timestamp_raw = std.time.milliTimestamp();
    const timestamp: u64 = if (timestamp_raw < 0) 0 else @intCast(timestamp_raw);

    var pid_component: u64 = 0;
    if (builtin.os.tag == .linux) {
        pid_component = @intCast(std.os.linux.getpid());
    } else if (builtin.os.tag == .macos or builtin.os.tag == .freebsd or builtin.os.tag == .netbsd or builtin.os.tag == .openbsd) {
        pid_component = @intCast(std.c.getpid());
    }

    var entropy_buf: [32]u8 = undefined;
    std.crypto.random.bytes(&entropy_buf);

    var hasher = std.crypto.hash.Blake2b256.init(.{});
    hasher.update(&entropy_buf);
    hasher.update(std.mem.asBytes(&timestamp));
    hasher.update(std.mem.asBytes(&pid_component));
    const digest = hasher.finalResult();

    return mem.readInt(u64, digest[0..8], .little);
}

fn mixHash(h: u64) u64 {
    var mixed = h ^ (h >> IoConfig.MIX_SHIFT);
    mixed *%= IoConfig.WYHASH_PRIME_1;
    mixed ^= mixed >> IoConfig.MIX_SHIFT;
    mixed *%= IoConfig.WYHASH_PRIME_2;
    return mixed ^ (mixed >> IoConfig.MIX_SHIFT);
}

pub const MMAP = struct {
    file: fs.File,
    buffer: ?[]align(IoConfig.PAGE_SIZE) u8,
    allocator: Allocator,
    is_writable: bool,
    actual_size: usize,

    pub fn open(allocator: Allocator, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try fs.cwd().openFile(path, mode);
        errdefer file.close();

        const stat = try file.stat();
        const size_u64: u64 = stat.size;
        if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
        var size: usize = @intCast(size_u64);

        const is_writable = mode.mode == .read_write or mode.mode == .write_only;

        var prot_flags: u32 = std.posix.PROT.READ;
        if (is_writable and mode.mode == .read_write) {
            prot_flags |= std.posix.PROT.WRITE;
        } else if (is_writable and mode.mode == .write_only) {
            prot_flags = std.posix.PROT.WRITE;
        }

        if (size == 0) {
            if (is_writable) {
                try file.setEndPos(IoConfig.PAGE_SIZE);
                var zeros: [IoConfig.PAGE_SIZE]u8 = undefined;
                @memset(&zeros, 0);
                try file.pwriteAll(&zeros, 0);
                size = IoConfig.PAGE_SIZE;
            } else {
                return error.FileIsEmpty;
            }
        }

        const aligned_size = std.mem.alignForward(usize, size, IoConfig.PAGE_SIZE);

        const map_type: std.posix.MAP.TYPE = if (is_writable) .SHARED else .PRIVATE;

        const buffer = try std.posix.mmap(
            null,
            aligned_size,
            prot_flags,
            .{ .TYPE = map_type },
            file.handle,
            0
        );

        return .{
            .file = file,
            .buffer = buffer,
            .allocator = allocator,
            .is_writable = is_writable,
            .actual_size = size,
        };
    }

    pub fn openWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try dir.openFile(path, mode);
        errdefer file.close();

        const stat = try file.stat();
        const size_u64: u64 = stat.size;
        if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
        var size: usize = @intCast(size_u64);

        const is_writable = mode.mode == .read_write or mode.mode == .write_only;

        var prot_flags: u32 = std.posix.PROT.READ;
        if (is_writable and mode.mode == .read_write) {
            prot_flags |= std.posix.PROT.WRITE;
        } else if (is_writable and mode.mode == .write_only) {
            prot_flags = std.posix.PROT.WRITE;
        }

        if (size == 0) {
            if (is_writable) {
                try file.setEndPos(IoConfig.PAGE_SIZE);
                var zeros: [IoConfig.PAGE_SIZE]u8 = undefined;
                @memset(&zeros, 0);
                try file.pwriteAll(&zeros, 0);
                size = IoConfig.PAGE_SIZE;
            } else {
                return error.FileIsEmpty;
            }
        }

        const aligned_size = std.mem.alignForward(usize, size, IoConfig.PAGE_SIZE);
        const map_type: std.posix.MAP.TYPE = if (is_writable) .SHARED else .PRIVATE;

        const buffer = try std.posix.mmap(
            null,
            aligned_size,
            prot_flags,
            .{ .TYPE = map_type },
            file.handle,
            0
        );

        return .{
            .file = file,
            .buffer = buffer,
            .allocator = allocator,
            .is_writable = is_writable,
            .actual_size = size,
        };
    }

    pub fn close(self: *MMAP) void {
        if (self.buffer) |buf| {
            std.posix.munmap(buf);
            self.buffer = null;
        }
        self.file.close();
    }

    pub fn read(self: *const MMAP, offset: usize, len: usize) ![]const u8 {
        const buf = self.buffer orelse return error.BufferNotMapped;
        if (offset >= buf.len) return error.OutOfBounds;
        if (len > buf.len) return error.OutOfBounds;
        if (offset > buf.len - len) {
            return buf[offset..buf.len];
        }
        return buf[offset..offset + len];
    }

    pub const SyncMode = enum {
        sync,
        nosync,
    };

    pub fn write(self: *MMAP, offset: usize, data: []const u8, sync_mode: SyncMode) !void {
        const buf = self.buffer orelse return error.BufferNotMapped;
        if (offset > buf.len) return error.OutOfBounds;
        if (data.len > buf.len - offset) return error.OutOfBounds;
        @memcpy(buf[offset..offset + data.len], data);
        const should_sync = sync_mode == .sync;
        try std.posix.msync(buf, .{ .SYNC = should_sync });
    }

    pub fn append(self: *MMAP, data: []const u8) !void {
        const buf = self.buffer orelse return error.BufferNotMapped;

        const stat = try self.file.stat();
        const current_size: usize = @intCast(stat.size);
        const new_size = current_size + data.len;

        const new_buf = blk: {
            std.posix.munmap(buf);
            self.buffer = null;

            try self.file.setEndPos(new_size);
            try self.file.pwriteAll(data, current_size);

            const aligned_size = std.mem.alignForward(usize, new_size, IoConfig.PAGE_SIZE);

            break :blk try std.posix.mmap(
                null,
                aligned_size,
                std.posix.PROT.READ | std.posix.PROT.WRITE,
                .{ .TYPE = .SHARED },
                self.file.handle,
                0
            );
        };

        self.buffer = new_buf;
        self.actual_size = new_size;
    }
};

pub const DurableWriter = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    allocator: Allocator,
    flush_depth: usize,
    enable_sync: bool,

    pub fn init(allocator: Allocator, path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try fs.cwd().createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });

        return .{
            .file = file,
            .allocator = allocator,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .flush_depth = 0,
            .enable_sync = enable_sync,
        };
    }

    pub fn initWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try dir.createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });

        return .{
            .file = file,
            .allocator = allocator,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .flush_depth = 0,
            .enable_sync = enable_sync,
        };
    }

    pub fn deinit(self: *DurableWriter) !void {
        try self.flush();
        if (self.enable_sync) {
            try self.file.sync();
        }
        self.file.close();
    }

    pub fn deinitNoError(self: *DurableWriter) void {
        self.flush() catch {};
        if (self.enable_sync) {
            self.file.sync() catch {};
        }
        self.file.close();
    }

    pub fn write(self: *DurableWriter, data: []const u8) !void {
        if (self.pos == self.buffer.len) {
            try self.flush();
        }

        if (data.len >= self.buffer.len - self.pos) {
            if (self.pos > 0) {
                try self.flush();
            }
            if (data.len >= self.buffer.len) {
                var written: usize = 0;
                while (written < data.len) {
                    const n = try self.file.write(data[written..]);
                    if (n == 0) return error.EndOfStream;
                    written += n;
                }
                return;
            }
        }

        const space = self.buffer.len - self.pos;
        const to_copy = @min(data.len, space);
        @memcpy(self.buffer[self.pos..self.pos + to_copy], data[0..to_copy]);
        self.pos += to_copy;

        if (to_copy < data.len) {
            try self.flush();
            const remaining = data[to_copy..];
            @memcpy(self.buffer[0..remaining.len], remaining);
            self.pos = remaining.len;
        }
    }

    pub fn flush(self: *DurableWriter) !void {
        if (self.flush_depth > IoConfig.MAX_FLUSH_DEPTH) {
            return error.RecursionDepthExceeded;
        }
        self.flush_depth += 1;
        defer self.flush_depth -= 1;

        if (self.pos > 0) {
            var written: usize = 0;
            while (written < self.pos) {
                const n = try self.file.write(self.buffer[written..self.pos]);
                if (n == 0) return error.EndOfStream;
                written += n;
            }
            self.pos = 0;
        }
    }

    pub fn writeAll(self: *DurableWriter, data: []const u8) !void {
        try self.write(data);
        try self.flush();
    }
};

pub const BufferedReader = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    limit: usize,
    allocator: Allocator,
    max_read_bytes: usize,

    pub fn init(allocator: Allocator, path: []const u8) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .allocator = allocator,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = IoConfig.MAX_READ_BYTES,
        };
    }

    pub fn initWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8) !BufferedReader {
        const file = try dir.openFile(path, .{});
        return .{
            .file = file,
            .allocator = allocator,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = IoConfig.MAX_READ_BYTES,
        };
    }

    pub fn initWithMaxBytes(allocator: Allocator, path: []const u8, max_bytes: usize) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .allocator = allocator,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = max_bytes,
        };
    }

    pub fn deinit(self: *BufferedReader) void {
        self.file.close();
    }

    pub fn read(self: *BufferedReader, buf: []u8) !usize {
        var total: usize = 0;
        while (total < buf.len) {
            if (self.pos < self.limit) {
                const avail = @min(self.limit - self.pos, buf.len - total);
                @memcpy(buf[total..total + avail], self.buffer[self.pos..self.pos + avail]);
                self.pos += avail;
                total += avail;
            } else {
                const n = try self.file.read(self.buffer[0..]);
                self.limit = n;
                self.pos = 0;
                if (n == 0) break;
            }
        }
        return total;
    }

    pub fn readUntil(self: *BufferedReader, delim: u8, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();

        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, delim)) |idx| {
                    try list.appendSlice(chunk[0..idx + 1]);
                    self.pos += idx + 1;
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                const n = try self.file.read(self.buffer[0..]);
                self.limit = n;
                self.pos = 0;
                if (n == 0) return list.toOwnedSlice();
            }
        }
        return error.MaxBytesExceeded;
    }

    pub fn peek(self: *BufferedReader) !?u8 {
        if (self.pos < self.limit) return self.buffer[self.pos];
        const n = try self.file.read(self.buffer[0..]);
        self.limit = n;
        self.pos = 0;
        if (n == 0) return null;
        return self.buffer[0];
    }
};

pub const BufferedWriter = struct {
    file: fs.File,
    buffer: []u8,
    pos: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, file: fs.File, buffer_size: usize) !BufferedWriter {
        const buffer = try allocator.alloc(u8, buffer_size);
        errdefer allocator.free(buffer);
        return .{
            .file = file,
            .buffer = buffer,
            .pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferedWriter) !void {
        try self.flush();
        self.allocator.free(self.buffer);
    }

    pub fn deinitNoError(self: *BufferedWriter) void {
        self.flush() catch {};
        self.allocator.free(self.buffer);
    }

    pub fn writeByte(self: *BufferedWriter, byte: u8) !void {
        if (self.pos >= self.buffer.len) {
            try self.flush();
        }
        self.buffer[self.pos] = byte;
        self.pos += 1;
    }

    pub fn writeBytes(self: *BufferedWriter, data: []const u8) !void {
        if (data.len >= self.buffer.len - self.pos) {
            if (self.pos > 0) {
                try self.flush();
            }
            if (data.len >= self.buffer.len) {
                var written: usize = 0;
                while (written < data.len) {
                    const to_write = @min(IoConfig.LARGE_CHUNK_SIZE, data.len - written);
                    try self.file.writeAll(data[written..written + to_write]);
                    written += to_write;
                }
                return;
            }
        }

        var written: usize = 0;
        while (written < data.len) {
            const available = self.buffer.len - self.pos;
            const to_write = @min(available, data.len - written);

            @memcpy(self.buffer[self.pos..self.pos + to_write], data[written..written + to_write]);
            self.pos += to_write;
            written += to_write;

            if (self.pos >= self.buffer.len) {
                try self.flush();
            }
        }
    }

    pub fn flush(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }
};

pub fn stableHash(data: []const u8, seed: u64) u64 {
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

pub fn hash64(data: []const u8) u64 {
    const seed = generateRuntimeSeed();
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

pub fn hash32(data: []const u8) u32 {
    const h64 = hash64(data);
    const mixed = h64 ^ (h64 >> IoConfig.TRUNCATE_SHIFT);
    return @truncate(mixed);
}

pub fn pathJoin(allocator: Allocator, parts: []const []const u8) ![]u8 {
    if (parts.len == 0) {
        const empty: []u8 = &.{};
        return allocator.dupe(u8, empty);
    }

    const separator: u8 = if (builtin.os.tag == .windows) '\\' else '/';

    var total_len: usize = 0;
    var non_empty_count: usize = 0;
    var starts_with_slash = false;

    var i: usize = 0;
    while (i < parts.len) : (i += 1) {
        const part = parts[i];
        if (part.len > 0) {
            if (i == 0 and part[0] == '/') {
                starts_with_slash = true;
            }
            total_len += part.len;
            non_empty_count += 1;
        }
    }

    if (non_empty_count == 0) {
        const empty: []u8 = &.{};
        return allocator.dupe(u8, empty);
    }

    const sep_count = if (non_empty_count > 1) non_empty_count - 1 else 0;
    total_len += sep_count;

    const path = try allocator.alloc(u8, total_len);
    errdefer allocator.free(path);

    var pos: usize = 0;
    var is_first = true;
    for (parts) |part| {
        if (part.len == 0) continue;

        if (!is_first) {
            path[pos] = separator;
            pos += 1;
        }

        var skip_leading_slash = false;
        if (is_first and starts_with_slash and part.len > 0 and part[0] == '/') {
            skip_leading_slash = false;
        }

        const src = if (skip_leading_slash and part.len > 0 and part[0] == '/')
            part[1..]
        else
            part;

        @memcpy(path[pos..pos + src.len], src);
        pos += src.len;
        is_first = false;
    }
    return path;
}

pub fn pathExists(path: []const u8) bool {
    _ = fs.cwd().statFile(path) catch return false;
    return true;
}

pub fn pathExistsWithAccess(path: []const u8) !bool {
    _ = fs.cwd().statFile(path) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    return true;
}

pub fn createDirRecursive(allocator: Allocator, path: []const u8) !void {
    if (path.len == 0) return;

    const separator: u8 = if (builtin.os.tag == .windows) '\\' else '/';

    var it = if (builtin.os.tag == .windows)
        mem.splitAny(u8, path, "/\\")
    else
        mem.splitScalar(u8, path, separator);

    var current_list = std.ArrayList(u8).init(allocator);
    defer current_list.deinit();

    var first = true;
    while (it.next()) |part| {
        if (part.len == 0) {
            if (first and path.len > 0 and path[0] == separator) {
                try current_list.append(separator);
            }
            first = false;
            continue;
        }

        if (current_list.items.len > 0 and current_list.items[current_list.items.len - 1] != separator) {
            try current_list.append(separator);
        }
        try current_list.appendSlice(part);

        fs.cwd().makeDir(current_list.items) catch |err| {
            if (err == error.PathAlreadyExists) {
                const stat_result = fs.cwd().statFile(current_list.items) catch {
                    first = false;
                    continue;
                };
                if (stat_result.kind != .directory) {
                    return error.NotADirectory;
                }
                first = false;
                continue;
            }
            return err;
        };
        first = false;
    }
}

pub fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size == 0) {
        const empty: []u8 = &.{};
        return allocator.dupe(u8, empty);
    }
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return error.UnexpectedEndOfFile;
    return buf;
}

pub fn readFileWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8) ![]u8 {
    const file = try dir.openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size == 0) {
        const empty: []u8 = &.{};
        return allocator.dupe(u8, empty);
    }
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return error.UnexpectedEndOfFile;
    return buf;
}

pub fn readFileLimited(allocator: Allocator, path: []const u8, max_size: usize) ![]u8 {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size > max_size) return error.FileTooLarge;
    if (size == 0) {
        const empty: []u8 = &.{};
        return allocator.dupe(u8, empty);
    }
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return error.UnexpectedEndOfFile;
    return buf;
}

pub const WriteFileOptions = struct {
    create_backup: bool = false,
    sync_after_write: bool = false,
};

pub fn writeFile(path: []const u8, data: []const u8) !void {
    return writeFileWithOptions(path, data, .{});
}

pub fn writeFileWithOptions(path: []const u8, data: []const u8, options: WriteFileOptions) !void {
    if (options.create_backup) {
        const stat_result = fs.cwd().statFile(path) catch |err| {
            if (err != error.FileNotFound) return err;
            const file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
            defer file.close();
            try file.writeAll(data);
            if (options.sync_after_write) try file.sync();
            return;
        };
        _ = stat_result;

        var backup_buf: [4096]u8 = undefined;
        const backup_path = std.fmt.bufPrint(&backup_buf, "{s}.bak", .{path}) catch return error.InvalidPath;

        fs.cwd().copyFile(path, fs.cwd(), backup_path, .{}) catch |copy_err| {
            return copy_err;
        };
    }
    const file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    defer file.close();
    try file.writeAll(data);
    if (options.sync_after_write) try file.sync();
}

pub fn appendFile(path: []const u8, data: []const u8) !void {
    const file = fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| {
        if (err == error.FileNotFound) {
            const new_file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
            defer new_file.close();
            try new_file.writeAll(data);
            return;
        }
        return err;
    };
    defer file.close();
    try file.seekFromEnd(0);
    try file.writeAll(data);
}

pub fn deleteFile(path: []const u8) !void {
    const stat_result = try fs.cwd().statFile(path);
    if (stat_result.kind == .directory) {
        return fs.cwd().deleteTree(path);
    }
    try fs.cwd().deleteFile(path);
}

pub const CopyProgress = struct {
    bytes_copied: usize,
    total_bytes: usize,
};

pub fn copyFile(allocator: Allocator, src: []const u8, dst: []const u8) !void {
    return copyFileWithProgress(allocator, src, dst, null);
}

pub fn copyFileWithProgress(
    allocator: Allocator,
    src: []const u8,
    dst: []const u8,
    progress_callback: ?*const fn(CopyProgress) void
) !void {
    const src_file = try fs.cwd().openFile(src, .{});
    defer src_file.close();

    const dst_file = try fs.cwd().createFile(dst, .{ .mode = IoConfig.SECURE_FILE_MODE });
    defer dst_file.close();

    const stat = try src_file.stat();
    const total_size: usize = @intCast(stat.size);

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    var bytes_copied: usize = 0;
    while (true) {
        const n = try src_file.read(buffer);
        if (n == 0) break;
        try dst_file.writeAll(buffer[0..n]);
        bytes_copied += n;
        if (progress_callback) |cb| {
            cb(.{ .bytes_copied = bytes_copied, .total_bytes = total_size });
        }
    }
}

pub fn moveFile(allocator: Allocator, old: []const u8, new: []const u8) !void {
    fs.cwd().rename(old, new) catch |err| {
        if (err == error.RenameAcrossMountPoints or err == error.NotSameFileSystem) {
            try copyFile(allocator, old, new);
            fs.cwd().deleteFile(old) catch |del_err| {
                fs.cwd().deleteFile(new) catch {};
                return del_err;
            };
            return;
        }
        return err;
    };
}

pub fn listDir(allocator: Allocator, path: []const u8) ![][]u8 {
    var dir = try fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var list = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit();
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try list.append(name);
    }
    return list.toOwnedSlice();
}

pub fn createDir(allocator: Allocator, path: []const u8) !void {
    try createDirRecursive(allocator, path);
}

pub fn removeDir(path: []const u8) !void {
    const stat_result = try fs.cwd().statFile(path);
    if (stat_result.kind == .sym_link) {
        try fs.cwd().deleteFile(path);
        return;
    }
    try fs.cwd().deleteTree(path);
}

pub fn removeFile(path: []const u8) !void {
    try fs.cwd().deleteFile(path);
}

pub fn renameFile(old: []const u8, new: []const u8) !void {
    try fs.cwd().rename(old, new);
}

pub fn getFileSize(path: []const u8) !usize {
    const stat = try fs.cwd().statFile(path);
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(size_u64);
}

pub fn isDir(path: []const u8) bool {
    const stat_result = fs.cwd().statFile(path) catch return false;
    return stat_result.kind == .directory;
}

pub fn isFile(path: []const u8) bool {
    const stat_result = fs.cwd().statFile(path) catch return false;
    return stat_result.kind == .file;
}

pub inline fn toLittleEndian(comptime T: type, value: T) T {
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .little => value,
        .big => @byteSwap(value),
    };
}

pub inline fn fromLittleEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    return mem.readInt(T, bytes, .little);
}

pub inline fn toBigEndian(comptime T: type, value: T) T {
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .little => @byteSwap(value),
        .big => value,
    };
}

pub inline fn fromBigEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    return mem.readInt(T, bytes, .big);
}

pub fn sequentialWrite(allocator: Allocator, path: []const u8, data: []const []const u8) !void {
    const file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    defer file.close();

    var writer = try BufferedWriter.init(allocator, file, IoConfig.LARGE_CHUNK_SIZE);
    defer writer.deinitNoError();

    for (data) |chunk| {
        try writer.writeBytes(chunk);
    }
    try writer.flush();
}

pub fn sequentialRead(allocator: Allocator, path: []const u8, chunk_callback: *const fn([]const u8) anyerror!void) !void {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    while (true) {
        const n = try file.read(buffer);
        if (n == 0) break;
        try chunk_callback(buffer[0..n]);
    }
}

pub fn atomicWrite(_: Allocator, path: []const u8, data: []const u8) !void {
    var temp_buf: [4096]u8 = undefined;
    const temp_path = std.fmt.bufPrint(&temp_buf, "{s}.tmp.{d}", .{path, generateRuntimeSeed()}) catch return error.InvalidPath;

    const file = try fs.cwd().createFile(temp_path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    errdefer {
        file.close();
        fs.cwd().deleteFile(temp_path) catch {};
    }

    try file.writeAll(data);
    try file.sync();

    if (builtin.os.tag != .windows) {
        if (std.fs.path.dirname(path)) |dir_path| {
            if (fs.cwd().openDir(dir_path, .{})) |dir| {
                defer dir.dir.close();
                dir.dir.sync() catch {};
            } else |_| {}
        }
    }

    file.close();

    try fs.cwd().rename(temp_path, path);
}

pub const FileCompareResult = enum {
    equal,
    different,
    first_not_found,
    second_not_found,
    both_not_found,
    read_error,
};

pub fn compareFiles(allocator: Allocator, path1: []const u8, path2: []const u8) FileCompareResult {
    const data1 = readFile(allocator, path1) catch |err| {
        if (err == error.FileNotFound) {
            const data2 = readFile(allocator, path2) catch |err2| {
                if (err2 == error.FileNotFound) return .both_not_found;
                return .read_error;
            };
            allocator.free(data2);
            return .first_not_found;
        }
        return .read_error;
    };
    defer allocator.free(data1);

    const data2 = readFile(allocator, path2) catch |err| {
        if (err == error.FileNotFound) return .second_not_found;
        return .read_error;
    };
    defer allocator.free(data2);

    if (data1.len != data2.len) return .different;
    if (mem.eql(u8, data1, data2)) return .equal;
    return .different;
}

pub fn compareFilesEqual(allocator: Allocator, path1: []const u8, path2: []const u8) bool {
    return compareFiles(allocator, path1, path2) == .equal;
}

test "MMAP open and close" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_mmap.bin", .{});
    try file.writeAll("test data for mmap");
    file.close();

    var mmap = try MMAP.openWithDir(gpa, tmp_dir.dir, "test_mmap.bin", .{ .mode = .read_only });
    defer mmap.close();

    const content = try mmap.read(0, 9);
    try std.testing.expectEqualStrings("test data", content);
}

test "DurableWriter with sync" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var writer = try DurableWriter.initWithDir(gpa, tmp_dir.dir, "test_durable.txt", false);
    try writer.writeAll("hello world");
    try writer.deinit();

    const content = try readFileWithDir(gpa, tmp_dir.dir, "test_durable.txt");
    defer gpa.free(content);
    try std.testing.expectEqualStrings("hello world", content);
}

test "BufferedReader zero init" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_buffered.txt", .{});
    try file.writeAll("line1\nline2\nline3");
    file.close();

    var reader = try BufferedReader.initWithDir(gpa, tmp_dir.dir, "test_buffered.txt");
    defer reader.deinit();

    const line1 = try reader.readUntil('\n', gpa);
    defer gpa.free(line1);
    try std.testing.expectEqualStrings("line1\n", line1);

    const line2 = try reader.readUntil('\n', gpa);
    defer gpa.free(line2);
    try std.testing.expectEqualStrings("line2\n", line2);

    const line3 = try reader.readUntil('\n', gpa);
    defer gpa.free(line3);
    try std.testing.expectEqualStrings("line3", line3);
}

test "Stable hash mixing" {
    const data = "test";
    const seed: u64 = 12345;
    const hash1 = stableHash(data, seed);
    const hash2 = stableHash(data, seed);
    const hash3 = stableHash(data, 67890);

    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}

test "Path join with leading slash" {
    var gpa = std.testing.allocator;
    const path1 = try pathJoin(gpa, &.{ "/a", "b", "c" });
    defer gpa.free(path1);

    const path2 = try pathJoin(gpa, &.{ "a", "b", "c" });
    defer gpa.free(path2);

    try std.testing.expect(path1[0] == '/');
}

test "Atomic write" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const path = "test_atomic.txt";
    var path_buf: [256]u8 = undefined;
    const full_path = try tmp_dir.dir.realpath(path, &path_buf);
    _ = full_path;

    try atomicWrite(gpa, path, "data");
    defer fs.cwd().deleteFile(path) catch {};
    const content = try readFile(gpa, path);
    defer gpa.free(content);
    try std.testing.expectEqualStrings("data", content);
}
