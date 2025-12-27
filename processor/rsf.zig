const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const Error = types.Error;

pub const RSFLayer = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_bias: Tensor,
    t_bias: Tensor,
    s_weight_grad: Tensor,
    t_weight_grad: Tensor,
    s_bias_grad: Tensor,
    t_bias_grad: Tensor,
    dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        const fan_in = @as(f32, @floatFromInt(dim));
        const fan_out = @as(f32, @floatFromInt(dim));
        const xavier_std = math.sqrt(2.0 / (fan_in + fan_out));

        const s_w = try Tensor.randomUniform(allocator, &.{ dim, dim }, -xavier_std, xavier_std, 42);
        const t_w = try Tensor.randomUniform(allocator, &.{ dim, dim }, -xavier_std, xavier_std, 43);
        const s_b = try Tensor.zeros(allocator, &.{ 1, dim });
        const t_b = try Tensor.zeros(allocator, &.{ 1, dim });

        const s_w_grad = try Tensor.zeros(allocator, &.{ dim, dim });
        const t_w_grad = try Tensor.zeros(allocator, &.{ dim, dim });
        const s_b_grad = try Tensor.zeros(allocator, &.{ 1, dim });
        const t_b_grad = try Tensor.zeros(allocator, &.{ 1, dim });

        return .{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_bias = s_b,
            .t_bias = t_b,
            .s_weight_grad = s_w_grad,
            .t_weight_grad = t_w_grad,
            .s_bias_grad = s_b_grad,
            .t_bias_grad = t_b_grad,
            .dim = dim,
            .allocator = allocator
        };
    }

    pub fn deinit(self: *RSFLayer) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        self.s_bias.deinit();
        self.t_bias.deinit();
        self.s_weight_grad.deinit();
        self.t_weight_grad.deinit();
        self.s_bias_grad.deinit();
        self.t_bias_grad.deinit();
    }

    pub fn zeroGradients(self: *RSFLayer) void {
        @memset(self.s_weight_grad.data, 0);
        @memset(self.t_weight_grad.data, 0);
        @memset(self.s_bias_grad.data, 0);
        @memset(self.t_bias_grad.data, 0);
    }

    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        var x2_t = try x2.transpose(&.{ 1, 0 });
        defer x2_t.deinit();
        var s_x2_t = try self.s_weight.matmul(&x2_t, self.allocator);
        defer s_x2_t.deinit();
        var s_x2 = try s_x2_t.transpose(&.{ 1, 0 });
        defer s_x2.deinit();
        try s_x2.add(&self.s_bias);
        try s_x2.clip(-5.0, 5.0);
        try s_x2.exp();
        try x1.mul(&s_x2);
        var x1_t = try x1.transpose(&.{ 1, 0 });
        defer x1_t.deinit();
        var t_y1_t = try self.t_weight.matmul(&x1_t, self.allocator);
        defer t_y1_t.deinit();
        var t_y1 = try t_y1_t.transpose(&.{ 1, 0 });
        defer t_y1.deinit();
        try t_y1.add(&self.t_bias);
        try x2.add(&t_y1);
    }

    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        var y1_t = try y1.transpose(&.{ 1, 0 });
        defer y1_t.deinit();
        var t_y1_t = try self.t_weight.matmul(&y1_t, self.allocator);
        defer t_y1_t.deinit();
        var t_y1 = try t_y1_t.transpose(&.{ 1, 0 });
        defer t_y1.deinit();
        try t_y1.add(&self.t_bias);
        try y2.sub(&t_y1);
        var y2_t = try y2.transpose(&.{ 1, 0 });
        defer y2_t.deinit();
        var s_y2_t = try self.s_weight.matmul(&y2_t, self.allocator);
        defer s_y2_t.deinit();
        var s_y2 = try s_y2_t.transpose(&.{ 1, 0 });
        defer s_y2.deinit();
        try s_y2.add(&self.s_bias);
        try s_y2.clip(-5.0, 5.0);
        try s_y2.exp();
        var log_s = try s_y2.copy(self.allocator);
        defer log_s.deinit();
        try log_s.log();
        try y1.div(&s_y2);
    }

    pub fn parameters(self: *const RSFLayer) [4]*Tensor {
        return .{ &self.s_weight, &self.t_weight, &self.s_bias, &self.t_bias };
    }
};

pub const RSF = struct {
    layers: []RSFLayer,
    num_layers: usize,
    dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize, num_layers: usize) !RSF {
        const layers = try allocator.alloc(RSFLayer, num_layers);
        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            layers[l] = try RSFLayer.init(allocator, dim);
        }
        return .{ .layers = layers, .num_layers = num_layers, .dim = dim, .allocator = allocator };
    }

    pub fn deinit(self: *RSF) void {
        var l: usize = 0;
        while (l < self.layers.len) : (l += 1) {
            self.layers[l].deinit();
        }
        self.allocator.free(self.layers);
    }

    pub fn forward(self: *RSF, x: *Tensor) !void {
        if (x.shape.dims.len != 2 or x.shape.dims[1] != self.dim * 2) return Error.ShapeMismatch;
        var x1 = try x.slice(&.{ 0, 0 }, &.{ x.shape.dims[0], self.dim });
        defer x1.deinit();
        var x2 = try x.slice(&.{ 0, self.dim }, &.{ x.shape.dims[0], self.dim * 2 });
        defer x2.deinit();
        var l: usize = 0;
        while (l < self.layers.len) : (l += 1) {
            try self.layers[l].forward(&x1, &x2);
        }
        @memcpy(x.data[0..x1.data.len], x1.data);
        @memcpy(x.data[x1.data.len..], x2.data);
    }

    pub fn inverse(self: *RSF, y: *Tensor) !void {
        if (y.shape.dims.len != 2 or y.shape.dims[1] != self.dim * 2) return Error.ShapeMismatch;
        var y1 = try y.slice(&.{ 0, 0 }, &.{ y.shape.dims[0], self.dim });
        defer y1.deinit();
        var y2 = try y.slice(&.{ 0, self.dim }, &.{ y.shape.dims[0], self.dim * 2 });
        defer y2.deinit();
        var idx = self.num_layers;
        while (idx > 0) : (idx -= 1) {
            try self.layers[idx - 1].inverse(&y1, &y2);
        }
        @memcpy(y.data[0..y1.data.len], y1.data);
        @memcpy(y.data[y1.data.len..], y2.data);
    }

    pub fn compose(self: *RSF, other: *const RSF) !void {
        if (self.dim != other.dim) return Error.ShapeMismatch;
        const new_num = self.num_layers + other.num_layers;
        const new_layers = try self.allocator.alloc(RSFLayer, new_num);
        @memcpy(new_layers[0..self.num_layers], self.layers);
        @memcpy(new_layers[self.num_layers..], other.layers);
        self.allocator.free(self.layers);
        self.layers = new_layers;
        self.num_layers = new_num;
    }

    pub fn gradientCheck(self: *RSF, x: *Tensor, eps: f32) !f32 {
        const y = try x.copy(self.allocator);
        defer y.deinit();
        try self.forward(&y);
        const y_fwd = try y.copy(self.allocator);
        defer y_fwd.deinit();
        const grad = try self.backward(&y, x);
        defer grad.deinit();
        var total_error: f32 = 0.0;
        var i: usize = 0;
        while (i < x.data.len) : (i += 1) {
            const orig = x.data[i];
            x.data[i] = orig + eps;
            try self.forward(&y);
            const fwd_plus = y.data[i];
            x.data[i] = orig - eps;
            try self.forward(&y);
            const fwd_minus = y.data[i];
            x.data[i] = orig;
            const num_grad = (fwd_plus - fwd_minus) / (2.0 * eps);
            const ana_grad = grad.data[i];
            total_error += math.fabs(num_grad - ana_grad);
        }
        return total_error / @as(f32, @floatFromInt(x.data.len));
    }

    pub fn backward(self: *RSF, grad_output: *const Tensor, x: *Tensor) !Tensor {
        if (grad_output.shape.dims.len != 2 or grad_output.shape.dims[1] != self.dim * 2) return Error.ShapeMismatch;

        for (self.layers) |*layer| {
            layer.zeroGradients();
        }

        var grad_x = try Tensor.zeros(self.allocator, grad_output.shape);
        errdefer grad_x.deinit();
        var current_grad = try grad_output.copy(self.allocator);
        defer current_grad.deinit();

        var saved_states = try self.allocator.alloc(struct { x1: Tensor, x2: Tensor, s_x2: Tensor, y1: Tensor }, self.num_layers);
        defer {
            var j: usize = 0;
            while (j < self.num_layers) : (j += 1) {
                saved_states[j].x1.deinit();
                saved_states[j].x2.deinit();
                saved_states[j].s_x2.deinit();
                saved_states[j].y1.deinit();
            }
            self.allocator.free(saved_states);
        }

        var x_copy = try x.copy(self.allocator);
        defer x_copy.deinit();
        var l: usize = 0;
        while (l < self.num_layers) : (l += 1) {
            var x1 = try x_copy.slice(&.{ 0, 0 }, &.{ x_copy.shape.dims[0], self.dim });
            var x2 = try x_copy.slice(&.{ 0, self.dim }, &.{ x_copy.shape.dims[0], self.dim * 2 });
            var x2_t = try x2.transpose(&.{ 1, 0 });
            defer x2_t.deinit();
            var s_x2_t = try self.layers[l].s_weight.matmul(&x2_t, self.allocator);
            defer s_x2_t.deinit();
            var s_x2 = try s_x2_t.transpose(&.{ 1, 0 });
            try s_x2.add(&self.layers[l].s_bias);
            try s_x2.clip(-5.0, 5.0);

            var y1_copy = try x1.copy(self.allocator);
            var s_exp = try s_x2.copy(self.allocator);
            try s_exp.exp();
            try y1_copy.mul(&s_exp);
            s_exp.deinit();

            saved_states[l] = .{
                .x1 = try x1.copy(self.allocator),
                .x2 = try x2.copy(self.allocator),
                .s_x2 = s_x2,
                .y1 = y1_copy,
            };
            x1.deinit();
            x2.deinit();

            var temp_x1 = try x_copy.slice(&.{ 0, 0 }, &.{ x_copy.shape[0], self.dim });
            defer temp_x1.deinit();
            var temp_x2 = try x_copy.slice(&.{ 0, self.dim }, &.{ x_copy.shape[0], self.dim * 2 });
            defer temp_x2.deinit();
            try self.layers[l].forward(&temp_x1, &temp_x2);
            @memcpy(x_copy.data[0..temp_x1.data.len], temp_x1.data);
            @memcpy(x_copy.data[temp_x1.data.len..], temp_x2.data);
        }

        var idx = self.num_layers;
        while (idx > 0) : (idx -= 1) {
            const layer = &self.layers[idx - 1];
            var g1 = try current_grad.slice(&.{ 0, 0 }, &.{ current_grad.shape[0], self.dim });
            defer g1.deinit();
            var g2 = try current_grad.slice(&.{ 0, self.dim }, &.{ current_grad.shape[0], self.dim * 2 });
            defer g2.deinit();

            var s_exp = try saved_states[idx - 1].s_x2.copy(self.allocator);
            defer s_exp.deinit();
            try s_exp.exp();

            var grad_pre_exp = try g1.copy(self.allocator);
            defer grad_pre_exp.deinit();
            try grad_pre_exp.mul(&saved_states[idx - 1].x1);
            try grad_pre_exp.mul(&s_exp);

            var grad_pre_exp_t = try grad_pre_exp.transpose(&.{ 1, 0 });
            defer grad_pre_exp_t.deinit();
            var s_weight_grad_t = try grad_pre_exp_t.matmul(&saved_states[idx - 1].x2, self.allocator);
            defer s_weight_grad_t.deinit();
            var s_weight_grad = try s_weight_grad_t.transpose(&.{ 1, 0 });
            defer s_weight_grad.deinit();

            var i: usize = 0;
            while (i < layer.s_weight_grad.data.len) : (i += 1) {
                layer.s_weight_grad.data[i] += s_weight_grad.data[i];
            }

            i = 0;
            while (i < layer.s_bias_grad.data.len) : (i += 1) {
                var sum: f32 = 0.0;
                var b: usize = 0;
                while (b < grad_pre_exp.shape[0]) : (b += 1) {
                    sum += grad_pre_exp.data[b * grad_pre_exp.shape[1] + i];
                }
                layer.s_bias_grad.data[i] += sum;
            }

            var g2_t = try g2.transpose(&.{ 1, 0 });
            defer g2_t.deinit();
            var t_weight_grad_t = try g2_t.matmul(&saved_states[idx - 1].y1, self.allocator);
            defer t_weight_grad_t.deinit();
            var t_weight_grad = try t_weight_grad_t.transpose(&.{ 1, 0 });
            defer t_weight_grad.deinit();

            i = 0;
            while (i < layer.t_weight_grad.data.len) : (i += 1) {
                layer.t_weight_grad.data[i] += t_weight_grad.data[i];
            }

            i = 0;
            while (i < layer.t_bias_grad.data.len) : (i += 1) {
                var sum: f32 = 0.0;
                var b: usize = 0;
                while (b < g2.shape[0]) : (b += 1) {
                    sum += g2.data[b * g2.shape[1] + i];
                }
                layer.t_bias_grad.data[i] += sum;
            }

            var grad_s = try g1.copy(self.allocator);
            defer grad_s.deinit();
            try grad_s.mul(&s_exp);

            var s_weight_t = try layer.s_weight.transpose(&.{ 1, 0 });
            defer s_weight_t.deinit();
            var grad_s_t = try grad_s.transpose(&.{ 1, 0 });
            defer grad_s_t.deinit();
            var backprop_s_t = try s_weight_t.matmul(&grad_s_t, self.allocator);
            defer backprop_s_t.deinit();
            var backprop_s = try backprop_s_t.transpose(&.{ 1, 0 });
            defer backprop_s.deinit();

            var t_weight_t = try layer.t_weight.transpose(&.{ 1, 0 });
            defer t_weight_t.deinit();
            var backprop_t_t = try t_weight_t.matmul(&g2_t, self.allocator);
            defer backprop_t_t.deinit();
            var backprop_t = try backprop_t_t.transpose(&.{ 1, 0 });
            defer backprop_t.deinit();

            var grad_x1 = try grad_x.slice(&.{ 0, 0 }, &.{ grad_x.shape[0], self.dim });
            defer grad_x1.deinit();
            var grad_x2 = try grad_x.slice(&.{ 0, self.dim }, &.{ grad_x.shape[0], self.dim * 2 });
            defer grad_x2.deinit();

            try grad_x1.add(&g1);
            try grad_x2.add(&backprop_s);
            try grad_x2.add(&backprop_t);
            try grad_x2.add(&g2);

            @memcpy(grad_x.data[0..grad_x1.data.len], grad_x1.data);
            @memcpy(grad_x.data[grad_x1.data.len..], grad_x2.data);

            const temp_grad = try grad_x.copy(self.allocator);
            current_grad.deinit();
            current_grad = temp_grad;
        }
        return grad_x;
    }

    pub fn save(self: *RSF, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();
        try writer.writeInt(usize, self.num_layers, .Little);
        try writer.writeInt(usize, self.dim, .Little);
        var l: usize = 0;
        while (l < self.layers.len) : (l += 1) {
            try self.layers[l].s_weight.save(writer);
            try self.layers[l].t_weight.save(writer);
            try self.layers[l].s_bias.save(writer);
            try self.layers[l].t_bias.save(writer);
        }
    }

    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();
        const num_layers = try reader.readInt(usize, .Little);
        const dim = try reader.readInt(usize, .Little);
        var rsf = try RSF.init(allocator, dim, num_layers);
        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            rsf.layers[l].s_weight = try Tensor.load(allocator, reader);
            rsf.layers[l].t_weight = try Tensor.load(allocator, reader);
            rsf.layers[l].s_bias = try Tensor.load(allocator, reader);
            rsf.layers[l].t_bias = try Tensor.load(allocator, reader);
        }
        return rsf;
    }

    pub fn invertibleCheck(self: *RSF, x: *Tensor, tol: f32) !bool {
        const y = try x.copy(self.allocator);
        defer y.deinit();
        try self.forward(&y);
        const x_rec = try y.copy(self.allocator);
        defer x_rec.deinit();
        try self.inverse(&x_rec);
        return try x.isClose(&x_rec, tol, tol);
    }

    pub fn layerNorm(self: *RSF, x: *Tensor, eps: f32) !void {
        const mean = try x.mean(self.allocator, 1);
        defer mean.deinit();
        const var_t = try x.variance(self.allocator, 1);
        defer var_t.deinit();
        var_t.addScalar(eps);
        var_t.sqrt();
        try x.sub(&mean);
        try x.div(&var_t);
    }
};

test "RSFLayer forward inverse" {
    const gpa = std.testing.allocator;
    var layer = try RSFLayer.init(gpa, 4);
    defer layer.deinit();
    var x1 = try Tensor.randomNormal(gpa, &.{ 2, 4 }, 0, 1, 42);
    defer x1.deinit();
    var x2 = try Tensor.randomNormal(gpa, &.{ 2, 4 }, 0, 1, 43);
    defer x2.deinit();
    var x_orig = try Tensor.concat(gpa, &.{ &x1, &x2 }, 1);
    defer x_orig.deinit();
    try layer.forward(&x1, &x2);
    var y1 = try x1.copy(gpa);
    defer y1.deinit();
    var y2 = try x2.copy(gpa);
    defer y2.deinit();
    try layer.inverse(&y1, &y2);
    try std.testing.expect(try x1.isClose(&y1, 1e-5, 1e-5));
    try std.testing.expect(try x2.isClose(&y2, 1e-5, 1e-5));
}

test "RSF compose" {
    const gpa = std.testing.allocator;
    var rsf1 = try RSF.init(gpa, 4, 2);
    defer rsf1.deinit();
    var rsf2 = try RSF.init(gpa, 4, 3);
    defer rsf2.deinit();
    try rsf1.compose(&rsf2);
    try std.testing.expectEqual(rsf1.num_layers, 5);
}

test "RSF gradient check" {
    const gpa = std.testing.allocator;
    var rsf = try RSF.init(gpa, 8, 3);
    defer rsf.deinit();
    var x = try Tensor.randomNormal(gpa, &.{ 10, 16 }, 0, 1, 44);
    defer x.deinit();
    const error_val = try rsf.gradientCheck(&x, 1e-5);
    try std.testing.expect(error_val < 1e-3);
}

test "RSF invertible" {
    const gpa = std.testing.allocator;
    var rsf = try RSF.init(gpa, 4, 5);
    defer rsf.deinit();
    var x = try Tensor.randomNormal(gpa, &.{ 5, 8 }, 0, 1, 45);
    defer x.deinit();
    try std.testing.expect(try rsf.invertibleCheck(&x, 1e-4));
}

test "RSF save load" {
    const gpa = std.testing.allocator;
    var rsf = try RSF.init(gpa, 4, 2);
    defer rsf.deinit();
    try rsf.save("test_rsf.bin");
    defer {
        std.fs.cwd().deleteFile("test_rsf.bin") catch |err| {
            std.log.warn("Failed to delete test file: {}", .{err});
        };
    }
    var rsf2 = try RSF.load(gpa, "test_rsf.bin");
    defer rsf2.deinit();
    try std.testing.expectEqual(rsf.num_layers, rsf2.num_layers);
}
