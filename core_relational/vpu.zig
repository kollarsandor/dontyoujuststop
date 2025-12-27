const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;

pub const VectorType = enum(u8) {
    f32x4 = 0,
    f32x8 = 1,
    f64x2 = 2,
    f64x4 = 3,
    i32x4 = 4,
    i32x8 = 5,

    pub fn toString(self: VectorType) []const u8 {
        return switch (self) {
            .f32x4 => "f32x4",
            .f32x8 => "f32x8",
            .f64x2 => "f64x2",
            .f64x4 => "f64x4",
            .i32x4 => "i32x4",
            .i32x8 => "i32x8",
        };
    }

    pub fn fromString(s: []const u8) ?VectorType {
        if (std.mem.eql(u8, s, "f32x4")) return .f32x4;
        if (std.mem.eql(u8, s, "f32x8")) return .f32x8;
        if (std.mem.eql(u8, s, "f64x2")) return .f64x2;
        if (std.mem.eql(u8, s, "f64x4")) return .f64x4;
        if (std.mem.eql(u8, s, "i32x4")) return .i32x4;
        if (std.mem.eql(u8, s, "i32x8")) return .i32x8;
        return null;
    }

    pub fn lanes(self: VectorType) usize {
        return switch (self) {
            .f32x4 => 4,
            .f32x8 => 8,
            .f64x2 => 2,
            .f64x4 => 4,
            .i32x4 => 4,
            .i32x8 => 8,
        };
    }

    pub fn elementSize(self: VectorType) usize {
        return switch (self) {
            .f32x4, .f32x8, .i32x4, .i32x8 => 4,
            .f64x2, .f64x4 => 8,
        };
    }

    pub fn totalSize(self: VectorType) usize {
        return self.lanes() * self.elementSize();
    }
};

pub fn SimdVector(comptime T: type, comptime N: usize) type {
    return struct {
        data: @Vector(N, T),

        const Self = @This();
        const VecType = @Vector(N, T);

        pub fn init(value: T) Self {
            return Self{ .data = @splat(value) };
        }

        pub fn initFromArray(arr: [N]T) Self {
            return Self{ .data = arr };
        }

        pub fn initFromSlice(slice: []const T) Self {
            var arr: [N]T = undefined;
            var i: usize = 0;
            while (i < N) : (i += 1) {
                if (i < slice.len) {
                    arr[i] = slice[i];
                } else {
                    arr[i] = 0;
                }
            }
            return Self{ .data = arr };
        }

        pub fn toArray(self: Self) [N]T {
            return self.data;
        }

        pub fn add(self: Self, other: Self) Self {
            return Self{ .data = self.data + other.data };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ .data = self.data - other.data };
        }

        pub fn mul(self: Self, other: Self) Self {
            return Self{ .data = self.data * other.data };
        }

        pub fn div(self: Self, other: Self) Self {
            return Self{ .data = self.data / other.data };
        }

        pub fn scale(self: Self, scalar: T) Self {
            const scalar_vec: VecType = @splat(scalar);
            return Self{ .data = self.data * scalar_vec };
        }

        pub fn dot(self: Self, other: Self) T {
            const product = self.data * other.data;
            return @reduce(.Add, product);
        }

        pub fn magnitude(self: Self) T {
            const squared = self.data * self.data;
            const sum = @reduce(.Add, squared);
            return @sqrt(sum);
        }

        pub fn normalize(self: Self) Self {
            const mag = self.magnitude();
            if (mag == 0) {
                return Self.init(0);
            }
            const mag_vec: VecType = @splat(mag);
            return Self{ .data = self.data / mag_vec };
        }

        pub fn fma(self: Self, mul_vec: Self, add_vec: Self) Self {
            return Self{ .data = @mulAdd(VecType, self.data, mul_vec.data, add_vec.data) };
        }

        pub fn sqrt(self: Self) Self {
            return Self{ .data = @sqrt(self.data) };
        }

        pub fn min(self: Self, other: Self) Self {
            return Self{ .data = @min(self.data, other.data) };
        }

        pub fn max(self: Self, other: Self) Self {
            return Self{ .data = @max(self.data, other.data) };
        }

        pub fn abs(self: Self) Self {
            const zero: VecType = @splat(0);
            const neg = zero - self.data;
            const is_negative = self.data < zero;
            return Self{ .data = @select(T, is_negative, neg, self.data) };
        }

        pub fn reduce_add(self: Self) T {
            return @reduce(.Add, self.data);
        }

        pub fn reduce_mul(self: Self) T {
            return @reduce(.Mul, self.data);
        }

        pub fn reduce_min(self: Self) T {
            return @reduce(.Min, self.data);
        }

        pub fn reduce_max(self: Self) T {
            return @reduce(.Max, self.data);
        }

        pub fn get(self: Self, index: usize) T {
            const arr = self.toArray();
            return arr[index];
        }

        pub fn set(self: *Self, index: usize, value: T) void {
            var arr = self.toArray();
            arr[index] = value;
            self.data = arr;
        }

        pub fn blend(self: Self, other: Self, mask: @Vector(N, bool)) Self {
            return Self{ .data = @select(T, mask, other.data, self.data) };
        }

        pub fn shuffle(self: Self, comptime indices: [N]i32) Self {
            return Self{ .data = @shuffle(T, self.data, undefined, indices) };
        }

        pub fn cross3(self: Self, other: Self) Self {
            if (N < 3) {
                return Self.init(0);
            }
            const a = self.toArray();
            const b = other.toArray();
            var result: [N]T = undefined;
            result[0] = a[1] * b[2] - a[2] * b[1];
            result[1] = a[2] * b[0] - a[0] * b[2];
            result[2] = a[0] * b[1] - a[1] * b[0];
            var i: usize = 3;
            while (i < N) : (i += 1) {
                result[i] = 0;
            }
            return Self.initFromArray(result);
        }

        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).magnitude();
        }

        pub fn lerp(self: Self, other: Self, t: T) Self {
            const one_minus_t: VecType = @splat(1 - t);
            const t_vec: VecType = @splat(t);
            return Self{ .data = self.data * one_minus_t + other.data * t_vec };
        }

        pub fn clamp(self: Self, min_val: T, max_val: T) Self {
            const min_vec: VecType = @splat(min_val);
            const max_vec: VecType = @splat(max_val);
            return Self{ .data = @max(min_vec, @min(max_vec, self.data)) };
        }

        pub fn negate(self: Self) Self {
            const zero: VecType = @splat(0);
            return Self{ .data = zero - self.data };
        }

        pub fn reflect(self: Self, normal: Self) Self {
            const d = self.dot(normal);
            const two_d: VecType = @splat(2 * d);
            return Self{ .data = self.data - two_d * normal.data };
        }
    };
}

pub const F32x4 = SimdVector(f32, 4);
pub const F32x8 = SimdVector(f32, 8);
pub const F64x2 = SimdVector(f64, 2);
pub const F64x4 = SimdVector(f64, 4);
pub const I32x4 = SimdVector(i32, 4);
pub const I32x8 = SimdVector(i32, 8);

pub const VectorBatchEntry = struct {
    vector_type: VectorType,
    data: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, vector_type: VectorType) !VectorBatchEntry {
        const size = vector_type.totalSize();
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);
        return VectorBatchEntry{
            .vector_type = vector_type,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VectorBatchEntry) void {
        self.allocator.free(self.data);
    }

    pub fn asF32x4(self: *const VectorBatchEntry) F32x4 {
        if (self.vector_type != .f32x4) return F32x4.init(0);
        const arr: *const [4]f32 = @ptrCast(@alignCast(self.data.ptr));
        return F32x4.initFromArray(arr.*);
    }

    pub fn asF32x8(self: *const VectorBatchEntry) F32x8 {
        if (self.vector_type != .f32x8) return F32x8.init(0);
        const arr: *const [8]f32 = @ptrCast(@alignCast(self.data.ptr));
        return F32x8.initFromArray(arr.*);
    }

    pub fn asF64x2(self: *const VectorBatchEntry) F64x2 {
        if (self.vector_type != .f64x2) return F64x2.init(0);
        const arr: *const [2]f64 = @ptrCast(@alignCast(self.data.ptr));
        return F64x2.initFromArray(arr.*);
    }

    pub fn asF64x4(self: *const VectorBatchEntry) F64x4 {
        if (self.vector_type != .f64x4) return F64x4.init(0);
        const arr: *const [4]f64 = @ptrCast(@alignCast(self.data.ptr));
        return F64x4.initFromArray(arr.*);
    }

    pub fn setF32x4(self: *VectorBatchEntry, vec: F32x4) void {
        if (self.vector_type != .f32x4) return;
        const arr = vec.toArray();
        const dest: *[4]f32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF32x8(self: *VectorBatchEntry, vec: F32x8) void {
        if (self.vector_type != .f32x8) return;
        const arr = vec.toArray();
        const dest: *[8]f32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF64x2(self: *VectorBatchEntry, vec: F64x2) void {
        if (self.vector_type != .f64x2) return;
        const arr = vec.toArray();
        const dest: *[2]f64 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF64x4(self: *VectorBatchEntry, vec: F64x4) void {
        if (self.vector_type != .f64x4) return;
        const arr = vec.toArray();
        const dest: *[4]f64 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }
};

pub const VectorBatch = struct {
    vectors: ArrayList(VectorBatchEntry),
    batch_size: usize,
    allocator: Allocator,
    processed_count: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, batch_size: usize) Self {
        return Self{
            .vectors = ArrayList(VectorBatchEntry).init(allocator),
            .batch_size = batch_size,
            .allocator = allocator,
            .processed_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var i: usize = 0;
        while (i < self.vectors.items.len) : (i += 1) {
            self.vectors.items[i].deinit();
        }
        self.vectors.deinit();
    }

    pub fn addVector(self: *Self, vector_type: VectorType) !usize {
        var entry = try VectorBatchEntry.init(self.allocator, vector_type);
        try self.vectors.append(entry);
        return self.vectors.items.len - 1;
    }

    pub fn addF32x4(self: *Self, vec: F32x4) !usize {
        const idx = try self.addVector(.f32x4);
        self.vectors.items[idx].setF32x4(vec);
        return idx;
    }

    pub fn addF32x8(self: *Self, vec: F32x8) !usize {
        const idx = try self.addVector(.f32x8);
        self.vectors.items[idx].setF32x8(vec);
        return idx;
    }

    pub fn addF64x2(self: *Self, vec: F64x2) !usize {
        const idx = try self.addVector(.f64x2);
        self.vectors.items[idx].setF64x2(vec);
        return idx;
    }

    pub fn addF64x4(self: *Self, vec: F64x4) !usize {
        const idx = try self.addVector(.f64x4);
        self.vectors.items[idx].setF64x4(vec);
        return idx;
    }

    pub fn getEntry(self: *Self, index: usize) ?*VectorBatchEntry {
        if (index >= self.vectors.items.len) return null;
        return &self.vectors.items[index];
    }

    pub fn processBatch(self: *Self, operation: BatchOperation) !void {
        var i: usize = 0;
        while (i < self.vectors.items.len) : (i += 1) {
            const entry = &self.vectors.items[i];
            switch (operation) {
                .normalize => try self.normalizeEntry(entry),
                .scale => |s| try self.scaleEntry(entry, s),
                .abs => try self.absEntry(entry),
                .sqrt => try self.sqrtEntry(entry),
            }
        }
        self.processed_count += self.vectors.items.len;
    }

    fn normalizeEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = entry.asF32x4();
                entry.setF32x4(vec.normalize());
            },
            .f32x8 => {
                const vec = entry.asF32x8();
                entry.setF32x8(vec.normalize());
            },
            .f64x2 => {
                const vec = entry.asF64x2();
                entry.setF64x2(vec.normalize());
            },
            .f64x4 => {
                const vec = entry.asF64x4();
                entry.setF64x4(vec.normalize());
            },
            else => {},
        }
    }

    fn scaleEntry(self: *Self, entry: *VectorBatchEntry, scalar: f64) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = entry.asF32x4();
                entry.setF32x4(vec.scale(@floatCast(scalar)));
            },
            .f32x8 => {
                const vec = entry.asF32x8();
                entry.setF32x8(vec.scale(@floatCast(scalar)));
            },
            .f64x2 => {
                const vec = entry.asF64x2();
                entry.setF64x2(vec.scale(scalar));
            },
            .f64x4 => {
                const vec = entry.asF64x4();
                entry.setF64x4(vec.scale(scalar));
            },
            else => {},
        }
    }

    fn absEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = entry.asF32x4();
                entry.setF32x4(vec.abs());
            },
            .f32x8 => {
                const vec = entry.asF32x8();
                entry.setF32x8(vec.abs());
            },
            .f64x2 => {
                const vec = entry.asF64x2();
                entry.setF64x2(vec.abs());
            },
            .f64x4 => {
                const vec = entry.asF64x4();
                entry.setF64x4(vec.abs());
            },
            else => {},
        }
    }

    fn sqrtEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = entry.asF32x4();
                entry.setF32x4(vec.sqrt());
            },
            .f32x8 => {
                const vec = entry.asF32x8();
                entry.setF32x8(vec.sqrt());
            },
            .f64x2 => {
                const vec = entry.asF64x2();
                entry.setF64x2(vec.sqrt());
            },
            .f64x4 => {
                const vec = entry.asF64x4();
                entry.setF64x4(vec.sqrt());
            },
            else => {},
        }
    }

    pub fn transformAll(self: *Self, transform_fn: *const fn (F64x4) F64x4) !void {
        var i: usize = 0;
        while (i < self.vectors.items.len) : (i += 1) {
            const entry = &self.vectors.items[i];
            if (entry.vector_type == .f64x4) {
                const vec = entry.asF64x4();
                entry.setF64x4(transform_fn(vec));
            }
        }
    }

    pub fn reduceAll(self: *Self) f64 {
        var total: f64 = 0;
        var i: usize = 0;
        while (i < self.vectors.items.len) : (i += 1) {
            const entry = &self.vectors.items[i];
            switch (entry.vector_type) {
                .f32x4 => total += @as(f64, entry.asF32x4().reduce_add()),
                .f32x8 => total += @as(f64, entry.asF32x8().reduce_add()),
                .f64x2 => total += entry.asF64x2().reduce_add(),
                .f64x4 => total += entry.asF64x4().reduce_add(),
                else => {},
            }
        }
        return total;
    }

    pub fn count(self: *const Self) usize {
        return self.vectors.items.len;
    }

    pub fn clear(self: *Self) void {
        var i: usize = 0;
        while (i < self.vectors.items.len) : (i += 1) {
            self.vectors.items[i].deinit();
        }
        self.vectors.clearRetainingCapacity();
    }
};

pub const BatchOperation = union(enum) {
    normalize: void,
    scale: f64,
    abs: void,
    sqrt: void,
};

pub const Matrix4x4 = struct {
    data: [4]F32x4,

    const Self = @This();

    pub fn identity() Self {
        return Self{
            .data = .{
                F32x4.initFromArray(.{ 1, 0, 0, 0 }),
                F32x4.initFromArray(.{ 0, 1, 0, 0 }),
                F32x4.initFromArray(.{ 0, 0, 1, 0 }),
                F32x4.initFromArray(.{ 0, 0, 0, 1 }),
            },
        };
    }

    pub fn zero() Self {
        return Self{
            .data = .{
                F32x4.init(0),
                F32x4.init(0),
                F32x4.init(0),
                F32x4.init(0),
            },
        };
    }

    pub fn fromRows(r0: [4]f32, r1: [4]f32, r2: [4]f32, r3: [4]f32) Self {
        return Self{
            .data = .{
                F32x4.initFromArray(r0),
                F32x4.initFromArray(r1),
                F32x4.initFromArray(r2),
                F32x4.initFromArray(r3),
            },
        };
    }

    pub fn get(self: Self, row_idx: usize, col_idx: usize) f32 {
        return self.data[row_idx].get(col_idx);
    }

    pub fn set(self: *Self, row_idx: usize, col_idx: usize, value: f32) void {
        self.data[row_idx].set(col_idx, value);
    }

    pub fn row(self: Self, idx: usize) F32x4 {
        return self.data[idx];
    }

    pub fn col(self: Self, idx: usize) F32x4 {
        return F32x4.initFromArray(.{
            self.data[0].get(idx),
            self.data[1].get(idx),
            self.data[2].get(idx),
            self.data[3].get(idx),
        });
    }

    pub fn add(self: Self, other: Self) Self {
        return Self{
            .data = .{
                self.data[0].add(other.data[0]),
                self.data[1].add(other.data[1]),
                self.data[2].add(other.data[2]),
                self.data[3].add(other.data[3]),
            },
        };
    }

    pub fn sub(self: Self, other: Self) Self {
        return Self{
            .data = .{
                self.data[0].sub(other.data[0]),
                self.data[1].sub(other.data[1]),
                self.data[2].sub(other.data[2]),
                self.data[3].sub(other.data[3]),
            },
        };
    }

    pub fn scale(self: Self, scalar: f32) Self {
        return Self{
            .data = .{
                self.data[0].scale(scalar),
                self.data[1].scale(scalar),
                self.data[2].scale(scalar),
                self.data[3].scale(scalar),
            },
        };
    }
};

pub const MatrixOps = struct {
    statistics: *VPUStatistics,

    const Self = @This();

    pub fn init(statistics: *VPUStatistics) Self {
        return Self{ .statistics = statistics };
    }

    pub fn matmul4x4(self: *Self, a: Matrix4x4, b: Matrix4x4) Matrix4x4 {
        var result = Matrix4x4.zero();
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                var sum: f32 = 0;
                var k: usize = 0;
                while (k < 4) : (k += 1) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        self.statistics.simd_instructions_used += 16;
        self.statistics.operations_completed += 1;
        return result;
    }

    pub fn matmul4x4Simd(self: *Self, a: Matrix4x4, b: Matrix4x4) Matrix4x4 {
        const b_t = self.transpose4x4(b);
        var result: [4]F32x4 = undefined;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const row_a = a.data[i];
            result[i] = F32x4.initFromArray(.{
                row_a.dot(b_t.data[0]),
                row_a.dot(b_t.data[1]),
                row_a.dot(b_t.data[2]),
                row_a.dot(b_t.data[3]),
            });
        }
        self.statistics.simd_instructions_used += 16;
        self.statistics.operations_completed += 1;
        return Matrix4x4{ .data = result };
    }

    pub fn transpose4x4(self: *Self, m: Matrix4x4) Matrix4x4 {
        var result = Matrix4x4.zero();
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                result.set(j, i, m.get(i, j));
            }
        }
        self.statistics.simd_instructions_used += 4;
        self.statistics.operations_completed += 1;
        return result;
    }

    pub fn determinant4x4(self: *Self, m: Matrix4x4) f32 {
        var det: f32 = 0;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const minor = self.minor3x3(m, 0, i);
            const sign: f32 = if (i % 2 == 0) 1.0 else -1.0;
            det += sign * m.get(0, i) * minor;
        }
        self.statistics.operations_completed += 1;
        return det;
    }

    fn minor3x3(self: *Self, m: Matrix4x4, row: usize, col: usize) f32 {
        _ = self;
        var submatrix: [3][3]f32 = undefined;
        var si: usize = 0;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            if (i == row) continue;
            var sj: usize = 0;
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                if (j == col) continue;
                submatrix[si][sj] = m.get(i, j);
                sj += 1;
            }
            si += 1;
        }
        return submatrix[0][0] * (submatrix[1][1] * submatrix[2][2] - submatrix[1][2] * submatrix[2][1]) -
            submatrix[0][1] * (submatrix[1][0] * submatrix[2][2] - submatrix[1][2] * submatrix[2][0]) +
            submatrix[0][2] * (submatrix[1][0] * submatrix[2][1] - submatrix[1][1] * submatrix[2][0]);
    }

    pub fn inverse4x4(self: *Self, m: Matrix4x4) ?Matrix4x4 {
        const det = self.determinant4x4(m);
        if (@fabs(det) < 1e-10) {
            return null;
        }

        var adjugate = Matrix4x4.zero();
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                const minor = self.minor3x3(m, i, j);
                const sign: f32 = if ((i + j) % 2 == 0) 1.0 else -1.0;
                adjugate.set(j, i, sign * minor);
            }
        }

        const inv_det = 1.0 / det;
        self.statistics.simd_instructions_used += 16;
        self.statistics.operations_completed += 1;
        return adjugate.scale(inv_det);
    }

    pub fn eigenvalues2x2(self: *Self, a: f32, b: f32, c: f32, d: f32) [2]Complex(f32) {
        const tr = a + d;
        const det = a * d - b * c;
        const discriminant = tr * tr - 4 * det;

        self.statistics.operations_completed += 1;

        if (discriminant >= 0) {
            const sqrt_disc = @sqrt(discriminant);
            return .{
                Complex(f32).init((tr + sqrt_disc) / 2, 0),
                Complex(f32).init((tr - sqrt_disc) / 2, 0),
            };
        } else {
            const sqrt_disc = @sqrt(-discriminant);
            return .{
                Complex(f32).init(tr / 2, sqrt_disc / 2),
                Complex(f32).init(tr / 2, -sqrt_disc / 2),
            };
        }
    }

    pub fn qr_decomposition(self: *Self, m: Matrix4x4) struct { q: Matrix4x4, r: Matrix4x4 } {
        var q = Matrix4x4.identity();
        var r = m;

        var col: usize = 0;
        while (col < 4) : (col += 1) {
            var v = r.col(col);
            var i: usize = 0;
            while (i < col) : (i += 1) {
                const qi = q.col(i);
                const proj = v.dot(qi);
                const proj_vec = qi.scale(proj);
                v = v.sub(proj_vec);
            }

            const norm = v.magnitude();
            if (@fabs(norm) > 1e-10) {
                v = v.scale(1.0 / norm);
            }

            var row: usize = 0;
            while (row < 4) : (row += 1) {
                q.set(row, col, v.get(row));
            }
        }

        r = self.matmul4x4(self.transpose4x4(q), m);

        self.statistics.simd_instructions_used += 32;
        self.statistics.operations_completed += 1;

        return .{ .q = q, .r = r };
    }

    pub fn trace(self: *Self, m: Matrix4x4) f32 {
        _ = self;
        return m.get(0, 0) + m.get(1, 1) + m.get(2, 2) + m.get(3, 3);
    }

    pub fn frobeniusNorm(self: *Self, m: Matrix4x4) f32 {
        var sum: f32 = 0;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const row_squared = m.data[i].mul(m.data[i]);
            sum += row_squared.reduce_add();
        }
        self.statistics.simd_instructions_used += 4;
        return @sqrt(sum);
    }
};

pub const RelationalVectorOps = struct {
    statistics: *VPUStatistics,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, statistics: *VPUStatistics) Self {
        return Self{
            .statistics = statistics,
            .allocator = allocator,
        };
    }

    pub fn computeEdgeVectorBatch(self: *Self, edges: []const Edge) !ArrayList(F64x4) {
        var result = ArrayList(F64x4).init(self.allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < edges.len) : (i += 1) {
            const edge = edges[i];
            const vec = F64x4.initFromArray(.{
                edge.weight,
                edge.quantum_correlation.re,
                edge.quantum_correlation.im,
                edge.fractal_dimension,
            });
            try result.append(vec);
            self.statistics.simd_instructions_used += 1;
        }

        self.statistics.operations_completed += 1;
        return result;
    }

    pub fn computeNodeSimilarity(self: *Self, node1: *const Node, node2: *const Node) f64 {
        const phase_diff = @fabs(node1.phase - node2.phase);
        const magnitude_diff = @fabs(node1.magnitude() - node2.magnitude());
        const quantum_inner_real = node1.quantum_state.re * node2.quantum_state.re +
            node1.quantum_state.im * node2.quantum_state.im;
        const quantum_inner_imag = node1.quantum_state.re * node2.quantum_state.im -
            node1.quantum_state.im * node2.quantum_state.re;
        const quantum_overlap = @sqrt(quantum_inner_real * quantum_inner_real +
            quantum_inner_imag * quantum_inner_imag);

        const v1 = F64x4.initFromArray(.{ phase_diff, magnitude_diff, quantum_overlap, 0 });
        const v2 = F64x4.initFromArray(.{ 0.3, 0.3, 0.4, 0 });
        const weighted = v1.mul(v2);
        const similarity = 1.0 - @min(weighted.reduce_add(), 1.0);

        self.statistics.simd_instructions_used += 2;
        self.statistics.operations_completed += 1;

        return similarity;
    }

    pub fn vectorizeGraph(self: *Self, graph: *SelfSimilarRelationalGraph) !ArrayList(F64x4) {
        var embeddings = ArrayList(F64x4).init(self.allocator);
        errdefer embeddings.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const embedding = F64x4.initFromArray(.{
                node.quantum_state.re,
                node.quantum_state.im,
                node.phase,
                node.magnitude(),
            });
            try embeddings.append(embedding);
            self.statistics.simd_instructions_used += 1;
        }

        self.statistics.operations_completed += 1;
        return embeddings;
    }

    pub fn parallelDotProduct(self: *Self, vectors_a: []const F64x4, vectors_b: []const F64x4) !ArrayList(f64) {
        var results = ArrayList(f64).init(self.allocator);
        errdefer results.deinit();

        const count = @min(vectors_a.len, vectors_b.len);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const dot = vectors_a[i].dot(vectors_b[i]);
            try results.append(dot);
            self.statistics.simd_instructions_used += 1;
        }

        self.statistics.operations_completed += 1;
        return results;
    }

    pub fn batchNormalize(self: *Self, vectors: []F64x4) void {
        var i: usize = 0;
        while (i < vectors.len) : (i += 1) {
            vectors[i] = vectors[i].normalize();
            self.statistics.simd_instructions_used += 1;
        }
        self.statistics.operations_completed += 1;
    }

    pub fn applyQuantumRotation(self: *Self, vector: F64x4, theta: f64, phi: f64) F64x4 {
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);
        const cos_phi = @cos(phi);
        const sin_phi = @sin(phi);

        const arr = vector.toArray();
        var result: [4]f64 = undefined;

        result[0] = arr[0] * cos_theta - arr[1] * sin_theta;
        result[1] = arr[0] * sin_theta + arr[1] * cos_theta;
        result[2] = arr[2] * cos_phi - arr[3] * sin_phi;
        result[3] = arr[2] * sin_phi + arr[3] * cos_phi;

        self.statistics.simd_instructions_used += 4;
        self.statistics.operations_completed += 1;

        return F64x4.initFromArray(result);
    }

    pub fn computeGraphLaplacian(self: *Self, adjacency: []const []const f64) !ArrayList(ArrayList(f64)) {
        const n = adjacency.len;
        var laplacian = ArrayList(ArrayList(f64)).init(self.allocator);
        errdefer {
            var i: usize = 0;
            while (i < laplacian.items.len) : (i += 1) {
                laplacian.items[i].deinit();
            }
            laplacian.deinit();
        }

        var i: usize = 0;
        while (i < n) : (i += 1) {
            var row = ArrayList(f64).init(self.allocator);
            var degree: f64 = 0;
            var j: usize = 0;
            while (j < n) : (j += 1) {
                degree += adjacency[i][j];
            }

            j = 0;
            while (j < n) : (j += 1) {
                if (i == j) {
                    try row.append(degree);
                } else {
                    try row.append(-adjacency[i][j]);
                }
            }
            try laplacian.append(row);
        }

        self.statistics.operations_completed += 1;
        return laplacian;
    }

    pub fn spectralEmbedding(self: *Self, vectors: []const F64x4, dimensions: usize) !ArrayList(F64x4) {
        _ = dimensions;
        var embedded = ArrayList(F64x4).init(self.allocator);
        errdefer embedded.deinit();

        var i: usize = 0;
        while (i < vectors.len) : (i += 1) {
            const normalized = vectors[i].normalize();
            try embedded.append(normalized);
            self.statistics.simd_instructions_used += 1;
        }

        self.statistics.operations_completed += 1;
        return embedded;
    }
};

pub const MemorySlice = struct {
    offset: usize,
    size: usize,
    in_use: bool,
};

pub const MemoryPool = struct {
    pool: []align(32) u8,
    free_list: ArrayList(MemorySlice),
    allocator: Allocator,
    total_allocated: usize,
    pool_size: usize,

    const Self = @This();
    const MIN_ALLOCATION_SIZE: usize = 32;

    pub fn init(allocator: Allocator, pool_size: usize) !Self {
        const aligned_size = std.mem.alignForward(usize, pool_size, 32);
        const pool = try allocator.alignedAlloc(u8, 32, aligned_size);
        @memset(pool, 0);

        var free_list = ArrayList(MemorySlice).init(allocator);
        try free_list.append(MemorySlice{
            .offset = 0,
            .size = aligned_size,
            .in_use = false,
        });

        return Self{
            .pool = pool,
            .free_list = free_list,
            .allocator = allocator,
            .total_allocated = 0,
            .pool_size = aligned_size,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.pool);
        self.free_list.deinit();
    }

    pub fn alloc(self: *Self, size: usize) ?[]align(32) u8 {
        const aligned_size = std.mem.alignForward(usize, @max(size, MIN_ALLOCATION_SIZE), 32);

        var i: usize = 0;
        while (i < self.free_list.items.len) : (i += 1) {
            var slice = &self.free_list.items[i];
            if (!slice.in_use and slice.size >= aligned_size) {
                if (slice.size > aligned_size + MIN_ALLOCATION_SIZE) {
                    const new_slice = MemorySlice{
                        .offset = slice.offset + aligned_size,
                        .size = slice.size - aligned_size,
                        .in_use = false,
                    };
                    slice.size = aligned_size;
                    self.free_list.insert(i + 1, new_slice) catch {};
                }

                slice.in_use = true;
                self.total_allocated += slice.size;

                const start = slice.offset;
                const end = start + aligned_size;
                const ptr: [*]align(32) u8 = @ptrCast(@alignCast(self.pool.ptr + start));
                return ptr[0 .. end - start];
            }
        }

        return null;
    }

    pub fn free(self: *Self, ptr: []align(32) u8) void {
        const ptr_addr = @intFromPtr(ptr.ptr);
        const pool_addr = @intFromPtr(self.pool.ptr);

        if (ptr_addr < pool_addr or ptr_addr >= pool_addr + self.pool_size) {
            return;
        }

        const offset = ptr_addr - pool_addr;

        var i: usize = 0;
        while (i < self.free_list.items.len) : (i += 1) {
            var slice = &self.free_list.items[i];
            if (slice.offset == offset and slice.in_use) {
                slice.in_use = false;
                self.total_allocated -= slice.size;
                @memset(ptr, 0);
                self.coalesceFreeBlocks();
                return;
            }
        }
    }

    fn coalesceFreeBlocks(self: *Self) void {
        if (self.free_list.items.len < 2) return;

        var i: usize = 0;
        while (i < self.free_list.items.len - 1) {
            const current = &self.free_list.items[i];
            const next = &self.free_list.items[i + 1];

            if (!current.in_use and !next.in_use and current.offset + current.size == next.offset) {
                current.size += next.size;
                _ = self.free_list.orderedRemove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub fn getAllocatedSize(self: *const Self) usize {
        return self.total_allocated;
    }

    pub fn getPoolSize(self: *const Self) usize {
        return self.pool_size;
    }

    pub fn getFreeSize(self: *const Self) usize {
        return self.pool_size - self.total_allocated;
    }

    pub fn reset(self: *Self) void {
        @memset(self.pool, 0);
        self.free_list.clearRetainingCapacity();
        self.free_list.append(MemorySlice{
            .offset = 0,
            .size = self.pool_size,
            .in_use = false,
        }) catch {};
        self.total_allocated = 0;
    }
};

pub const VPUStatistics = struct {
    operations_completed: usize,
    simd_instructions_used: usize,
    cache_hits: usize,
    cache_misses: usize,
    memory_allocated: usize,
    vectors_processed: usize,
    matrix_operations: usize,
    graph_operations: usize,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .operations_completed = 0,
            .simd_instructions_used = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .memory_allocated = 0,
            .vectors_processed = 0,
            .matrix_operations = 0,
            .graph_operations = 0,
        };
    }

    pub fn reset(self: *Self) void {
        self.operations_completed = 0;
        self.simd_instructions_used = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.memory_allocated = 0;
        self.vectors_processed = 0;
        self.matrix_operations = 0;
        self.graph_operations = 0;
    }

    pub fn clone(self: *const Self) Self {
        return Self{
            .operations_completed = self.operations_completed,
            .simd_instructions_used = self.simd_instructions_used,
            .cache_hits = self.cache_hits,
            .cache_misses = self.cache_misses,
            .memory_allocated = self.memory_allocated,
            .vectors_processed = self.vectors_processed,
            .matrix_operations = self.matrix_operations,
            .graph_operations = self.graph_operations,
        };
    }

    pub fn merge(self: *Self, other: *const Self) void {
        self.operations_completed += other.operations_completed;
        self.simd_instructions_used += other.simd_instructions_used;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.memory_allocated += other.memory_allocated;
        self.vectors_processed += other.vectors_processed;
        self.matrix_operations += other.matrix_operations;
        self.graph_operations += other.graph_operations;
    }

    pub fn getCacheHitRate(self: *const Self) f64 {
        const total = self.cache_hits + self.cache_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn getSimdEfficiency(self: *const Self) f64 {
        if (self.operations_completed == 0) return 0.0;
        return @as(f64, @floatFromInt(self.simd_instructions_used)) /
            @as(f64, @floatFromInt(self.operations_completed));
    }
};

pub const VectorCache = struct {
    entries: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    lru_queue: ArrayList(u64),
    statistics: *VPUStatistics,
    allocator: Allocator,

    const Self = @This();

    const CacheEntry = struct {
        data: []u8,
        vector_type: VectorType,
        timestamp: i64,
    };

    pub fn init(allocator: Allocator, max_entries: usize, statistics: *VPUStatistics) Self {
        return Self{
            .entries = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .max_entries = max_entries,
            .lru_queue = ArrayList(u64).init(allocator),
            .statistics = statistics,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.entries.deinit();
        self.lru_queue.deinit();
    }

    pub fn get(self: *Self, key: u64) ?CacheEntry {
        if (self.entries.get(key)) |entry| {
            self.statistics.cache_hits += 1;
            self.updateLRU(key);
            return entry;
        }
        self.statistics.cache_misses += 1;
        return null;
    }

    pub fn put(self: *Self, key: u64, data: []const u8, vector_type: VectorType) !void {
        if (self.entries.count() >= self.max_entries) {
            try self.evictLRU();
        }

        const data_copy = try self.allocator.dupe(u8, data);
        try self.entries.put(key, CacheEntry{
            .data = data_copy,
            .vector_type = vector_type,
            .timestamp = std.time.milliTimestamp(),
        });
        try self.lru_queue.append(key);
    }

    fn updateLRU(self: *Self, key: u64) void {
        var i: usize = 0;
        while (i < self.lru_queue.items.len) : (i += 1) {
            if (self.lru_queue.items[i] == key) {
                _ = self.lru_queue.orderedRemove(i);
                self.lru_queue.append(key) catch {};
                return;
            }
        }
    }

    fn evictLRU(self: *Self) !void {
        if (self.lru_queue.items.len == 0) return;
        const key = self.lru_queue.orderedRemove(0);
        if (self.entries.fetchRemove(key)) |removed| {
            self.allocator.free(removed.value.data);
        }
    }

    pub fn clear(self: *Self) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.entries.clearRetainingCapacity();
        self.lru_queue.clearRetainingCapacity();
    }
};

pub const VPU = struct {
    memory_pool: MemoryPool,
    vector_batch: VectorBatch,
    statistics: VPUStatistics,
    matrix_ops: MatrixOps,
    relational_ops: RelationalVectorOps,
    vector_cache: VectorCache,
    allocator: Allocator,
    cycle_count: usize,
    instruction_pointer: usize,

    const Self = @This();
    const DEFAULT_POOL_SIZE: usize = 1024 * 1024;
    const DEFAULT_BATCH_SIZE: usize = 256;
    const DEFAULT_CACHE_SIZE: usize = 1024;

    pub fn init(allocator: Allocator) !Self {
        return try initWithOptions(allocator, DEFAULT_POOL_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_CACHE_SIZE);
    }

    pub fn initWithOptions(
        allocator: Allocator,
        pool_size: usize,
        batch_size: usize,
        cache_size: usize,
    ) !Self {
        var statistics = VPUStatistics.init();
        var memory_pool = try MemoryPool.init(allocator, pool_size);
        errdefer memory_pool.deinit();

        var vector_batch = VectorBatch.init(allocator, batch_size);
        errdefer vector_batch.deinit();

        var matrix_ops = MatrixOps.init(&statistics);
        var relational_ops = RelationalVectorOps.init(allocator, &statistics);
        var vector_cache = VectorCache.init(allocator, cache_size, &statistics);

        return Self{
            .memory_pool = memory_pool,
            .vector_batch = vector_batch,
            .statistics = statistics,
            .matrix_ops = matrix_ops,
            .relational_ops = relational_ops,
            .vector_cache = vector_cache,
            .allocator = allocator,
            .cycle_count = 0,
            .instruction_pointer = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.memory_pool.deinit();
        self.vector_batch.deinit();
        self.vector_cache.deinit();
    }

    pub fn processVectors(self: *Self, operation: BatchOperation) !void {
        try self.vector_batch.processBatch(operation);
        self.statistics.vectors_processed += self.vector_batch.count();
        self.cycle_count += 1;
    }

    pub fn batchMatmul(self: *Self, matrices_a: []const Matrix4x4, matrices_b: []const Matrix4x4) !ArrayList(Matrix4x4) {
        var results = ArrayList(Matrix4x4).init(self.allocator);
        errdefer results.deinit();

        const count = @min(matrices_a.len, matrices_b.len);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const result = self.matrix_ops.matmul4x4Simd(matrices_a[i], matrices_b[i]);
            try results.append(result);
            self.statistics.matrix_operations += 1;
        }

        self.cycle_count += count;
        return results;
    }

    pub fn computeGraphEmbeddings(self: *Self, graph: *SelfSimilarRelationalGraph) !ArrayList(F64x4) {
        var embeddings = try self.relational_ops.vectorizeGraph(graph);
        self.relational_ops.batchNormalize(embeddings.items);
        self.statistics.graph_operations += 1;
        self.cycle_count += 1;
        return embeddings;
    }

    pub fn quantumVectorOps(self: *Self, vectors: []F64x4, theta: f64, phi: f64) void {
        var i: usize = 0;
        while (i < vectors.len) : (i += 1) {
            vectors[i] = self.relational_ops.applyQuantumRotation(vectors[i], theta, phi);
        }
        self.statistics.vectors_processed += vectors.len;
        self.cycle_count += 1;
    }

    pub fn addF32x4(self: *Self, vec: F32x4) !usize {
        const idx = try self.vector_batch.addF32x4(vec);
        self.statistics.memory_allocated += @sizeOf(f32) * 4;
        return idx;
    }

    pub fn addF64x4(self: *Self, vec: F64x4) !usize {
        const idx = try self.vector_batch.addF64x4(vec);
        self.statistics.memory_allocated += @sizeOf(f64) * 4;
        return idx;
    }

    pub fn computeSimilarityMatrix(self: *Self, embeddings: []const F64x4) !ArrayList(ArrayList(f64)) {
        var similarity_matrix = ArrayList(ArrayList(f64)).init(self.allocator);
        errdefer {
            var i: usize = 0;
            while (i < similarity_matrix.items.len) : (i += 1) {
                similarity_matrix.items[i].deinit();
            }
            similarity_matrix.deinit();
        }

        var i: usize = 0;
        while (i < embeddings.len) : (i += 1) {
            var row = ArrayList(f64).init(self.allocator);
            var j: usize = 0;
            while (j < embeddings.len) : (j += 1) {
                const dot = embeddings[i].dot(embeddings[j]);
                const mag_i = embeddings[i].magnitude();
                const mag_j = embeddings[j].magnitude();
                const similarity = if (mag_i * mag_j > 0) dot / (mag_i * mag_j) else 0;
                try row.append(similarity);
                self.statistics.simd_instructions_used += 3;
            }
            try similarity_matrix.append(row);
        }

        self.statistics.operations_completed += 1;
        self.cycle_count += 1;
        return similarity_matrix;
    }

    pub fn powerIteration(self: *Self, m: Matrix4x4, iterations: usize) F32x4 {
        var v = F32x4.initFromArray(.{ 1, 1, 1, 1 }).normalize();

        var iter: usize = 0;
        while (iter < iterations) : (iter += 1) {
            var result = F32x4.init(0);
            var i: usize = 0;
            while (i < 4) : (i += 1) {
                const row = m.row(i);
                const dot = row.dot(v);
                result.set(i, dot);
            }
            v = result.normalize();
            self.statistics.simd_instructions_used += 8;
        }

        self.statistics.operations_completed += 1;
        return v;
    }

    pub fn getStatistics(self: *const Self) VPUStatistics {
        return self.statistics.clone();
    }

    pub fn reset(self: *Self) void {
        self.statistics.reset();
        self.memory_pool.reset();
        self.vector_batch.clear();
        self.vector_cache.clear();
        self.cycle_count = 0;
        self.instruction_pointer = 0;
    }

    pub fn getCycleCount(self: *const Self) usize {
        return self.cycle_count;
    }

    pub fn getMemoryUsage(self: *const Self) usize {
        return self.memory_pool.getAllocatedSize();
    }

    pub fn allocSimdAligned(self: *Self, size: usize) ?[]align(32) u8 {
        const result = self.memory_pool.alloc(size);
        if (result != null) {
            self.statistics.memory_allocated += size;
        }
        return result;
    }

    pub fn freeSimdAligned(self: *Self, ptr: []align(32) u8) void {
        self.memory_pool.free(ptr);
    }
};

pub const LNSValue = struct {
    mantissa: f64,
    exponent: f64,
    sign: bool,

    const Self = @This();

    pub fn zero() Self {
        return Self{
            .mantissa = -std.math.inf(f64),
            .exponent = 0.0,
            .sign = true,
        };
    }

    pub fn fromFloat(value: f64) Self {
        if (value == 0.0) {
            return Self.zero();
        }
        const sign = value > 0.0;
        const abs_val = @fabs(value);
        return Self{
            .mantissa = @log(abs_val),
            .exponent = 1.0,
            .sign = sign,
        };
    }

    pub fn toFloat(self: Self) f64 {
        if (std.math.isInf(self.mantissa) and self.mantissa < 0.0) {
            return 0.0;
        }
        const magnitude = @exp(self.mantissa * self.exponent);
        return if (self.sign) magnitude else -magnitude;
    }

    pub fn add(self: Self, other: Self) Self {
        const f1 = self.toFloat();
        const f2 = other.toFloat();
        return Self.fromFloat(f1 + f2);
    }

    pub fn mul(self: Self, other: Self) Self {
        return Self{
            .mantissa = self.mantissa + other.mantissa,
            .exponent = self.exponent * other.exponent,
            .sign = self.sign == other.sign,
        };
    }

    pub fn div(self: Self, other: Self) Self {
        return Self{
            .mantissa = self.mantissa - other.mantissa,
            .exponent = self.exponent / other.exponent,
            .sign = self.sign == other.sign,
        };
    }
};

pub const LNSInstruction = union(enum) {
    lns_convolve: struct { src1: usize, src2: usize, dst: usize, kernel_size: usize },
    sparse_attention: struct { query: usize, key: usize, value: usize, dst: usize, sparsity: f64 },
    tensor_load: struct { addr: usize, dst: usize },
    tensor_store: struct { src: usize, addr: usize },
    lns_add: struct { src1: usize, src2: usize, dst: usize },
    lns_mul: struct { src1: usize, src2: usize, dst: usize },
    graph_transform: struct { node_id: usize, transform_type: usize },
    jump: struct { offset: isize },
    conditional_jump: struct { condition_reg: usize, offset: isize },
    halt: void,
};

test "SimdVector basic operations" {
    const v1 = F32x4.initFromArray(.{ 1, 2, 3, 4 });
    const v2 = F32x4.initFromArray(.{ 5, 6, 7, 8 });

    const sum = v1.add(v2);
    const sum_arr = sum.toArray();
    try std.testing.expectEqual(@as(f32, 6), sum_arr[0]);
    try std.testing.expectEqual(@as(f32, 8), sum_arr[1]);
    try std.testing.expectEqual(@as(f32, 10), sum_arr[2]);
    try std.testing.expectEqual(@as(f32, 12), sum_arr[3]);

    const dot = v1.dot(v2);
    try std.testing.expectEqual(@as(f32, 70), dot);

    const mag = v1.magnitude();
    try std.testing.expectApproxEqAbs(@as(f32, 5.477225575), mag, 0.0001);
}

test "Matrix4x4 operations" {
    var stats = VPUStatistics.init();
    var ops = MatrixOps.init(&stats);

    const m1 = Matrix4x4.identity();
    const m2 = Matrix4x4.identity();
    const result = ops.matmul4x4(m1, m2);

    try std.testing.expectEqual(@as(f32, 1), result.get(0, 0));
    try std.testing.expectEqual(@as(f32, 1), result.get(1, 1));
    try std.testing.expectEqual(@as(f32, 1), result.get(2, 2));
    try std.testing.expectEqual(@as(f32, 1), result.get(3, 3));
    try std.testing.expectEqual(@as(f32, 0), result.get(0, 1));
}

test "MemoryPool allocation" {
    const allocator = std.testing.allocator;
    var pool = try MemoryPool.init(allocator, 4096);
    defer pool.deinit();

    const ptr1 = pool.alloc(128);
    try std.testing.expect(ptr1 != null);
    try std.testing.expect(pool.getAllocatedSize() >= 128);

    const ptr2 = pool.alloc(256);
    try std.testing.expect(ptr2 != null);

    if (ptr1) |p1| {
        pool.free(p1);
    }

    try std.testing.expect(pool.getAllocatedSize() < pool.getPoolSize());
}

test "VPU initialization" {
    const allocator = std.testing.allocator;
    var vpu = try VPU.init(allocator);
    defer vpu.deinit();

    try std.testing.expect(vpu.cycle_count == 0);
    try std.testing.expect(vpu.statistics.operations_completed == 0);
}

test "VectorBatch operations" {
    const allocator = std.testing.allocator;
    var batch = VectorBatch.init(allocator, 16);
    defer batch.deinit();

    const v1 = F32x4.initFromArray(.{ 1, 2, 3, 4 });
    const idx = try batch.addF32x4(v1);
    try std.testing.expect(idx == 0);
    try std.testing.expect(batch.count() == 1);
}

test "LNSValue operations" {
    const v1 = LNSValue.fromFloat(2.0);
    const v2 = LNSValue.fromFloat(3.0);

    const product = v1.mul(v2);
    const result = product.toFloat();
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result, 0.0001);

    const sum = v1.add(v2);
    const sum_result = sum.toFloat();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), sum_result, 0.0001);
}
