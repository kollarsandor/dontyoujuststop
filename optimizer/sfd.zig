const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const Error = types.Error;
const testing = std.testing;

pub const LossFn = *const fn (params: *const Tensor, context: ?*anyopaque) anyerror!f32;

const FISHER_MAX: f32 = 1e6;
const WARMUP_STEPS: usize = 10;

pub const SFD = struct {
    fisher_diag: Tensor,
    momentum_buffer: Tensor,
    velocity_buffer: Tensor,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    clip_threshold: f32 = 1.0,
    step_count: usize = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_size: usize) !SFD {
        const diag = try Tensor.init(allocator, &.{param_size});
        const diag_data: [*]f32 = @ptrCast(@alignCast(diag.data.ptr));
        var i: usize = 0;
        while (i < param_size) : (i += 1) {
            diag_data[i] = 1.0;
        }
        const momentum = try Tensor.init(allocator, &.{param_size});
        @memset(momentum.data, 0);
        const velocity = try Tensor.init(allocator, &.{param_size});
        @memset(velocity.data, 0);
        return .{
            .fisher_diag = diag,
            .momentum_buffer = momentum,
            .velocity_buffer = velocity,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SFD) void {
        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.velocity_buffer.deinit();
    }

    pub fn update(self: *SFD, gradients: *const Tensor, params: *Tensor, lr: f32) !void {
        if (!gradients.shape.equals(&params.shape)) return Error.ShapeMismatch;
        self.step_count += 1;
        const grad_data: [*]const f32 = @ptrCast(@alignCast(gradients.data.ptr));
        const param_data: [*]f32 = @ptrCast(@alignCast(params.data.ptr));
        const momentum_data: [*]f32 = @ptrCast(@alignCast(self.momentum_buffer.data.ptr));
        const velocity_data: [*]f32 = @ptrCast(@alignCast(self.velocity_buffer.data.ptr));
        const fisher_data: [*]f32 = @ptrCast(@alignCast(self.fisher_diag.data.ptr));
        const param_count = gradients.data.len / @sizeOf(f32);
        const warmup_factor: f32 = if (self.step_count <= WARMUP_STEPS)
            @as(f32, @floatFromInt(self.step_count)) / @as(f32, @floatFromInt(WARMUP_STEPS))
        else
            1.0;
        var i: usize = 0;
        while (i < param_count) : (i += 1) {
            const g = grad_data[i];
            momentum_data[i] = self.beta1 * momentum_data[i] + (1.0 - self.beta1) * g;
            velocity_data[i] = self.beta2 * velocity_data[i] + (1.0 - self.beta2) * g * g;
            const m_hat = momentum_data[i] / (1.0 - math.pow(f32, self.beta1, @as(f32, @floatFromInt(self.step_count))));
            const v_hat = velocity_data[i] / (1.0 - math.pow(f32, self.beta2, @as(f32, @floatFromInt(self.step_count))));
            const adaptive_lr = lr * warmup_factor / (math.sqrt(v_hat) + self.eps);
            fisher_data[i] = self.beta1 * fisher_data[i] + (1.0 - self.beta1) * g * g;
            fisher_data[i] = @min(fisher_data[i], FISHER_MAX);
            const update_val = math.clamp(m_hat * adaptive_lr / (math.sqrt(fisher_data[i]) + self.eps), -self.clip_threshold, self.clip_threshold);
            param_data[i] -= update_val;
        }
    }

    pub fn adaptiveLR(self: *SFD, grad_norm: f32, param_norm: f32) f32 {
        return 1.0 / math.sqrt(grad_norm / (param_norm + self.eps) + self.eps);
    }

    pub fn spectralClip(self: *SFD, tensor: *Tensor, max_eig: f32) !void {
        const evals = try tensor.eigenvalues(self.allocator);
        defer evals.deinit();
        const max_ev_tensor = try evals.max(self.allocator, 0);
        defer max_ev_tensor.deinit();
        const max_ev_data: [*]const f32 = @ptrCast(@alignCast(max_ev_tensor.data.ptr));
        const max_ev: f32 = math.min(max_eig, max_ev_data[0]);
        if (max_ev > 0) {
            const scale = math.sqrt(max_eig / max_ev);
            const tensor_data: [*]f32 = @ptrCast(@alignCast(tensor.data.ptr));
            const tensor_count = tensor.data.len / @sizeOf(f32);
            var i: usize = 0;
            while (i < tensor_count) : (i += 1) {
                tensor_data[i] *= scale;
            }
        }
    }

    pub fn accumulateFisher(self: *SFD, grads: []const Tensor) !void {
        const fisher_data: [*]f32 = @ptrCast(@alignCast(self.fisher_diag.data.ptr));
        const fisher_count = self.fisher_diag.data.len / @sizeOf(f32);
        var i: usize = 0;
        while (i < grads.len) : (i += 1) {
            const g = grads[i];
            const g_data: [*]const f32 = @ptrCast(@alignCast(g.data.ptr));
            const g_count = g.data.len / @sizeOf(f32);
            var j: usize = 0;
            while (j < @min(fisher_count, g_count)) : (j += 1) {
                fisher_data[j] += g_data[j] * g_data[j];
                fisher_data[j] = @min(fisher_data[j], FISHER_MAX);
            }
        }
    }

    pub fn resetFisher(self: *SFD) void {
        const fisher_data: [*]f32 = @ptrCast(@alignCast(self.fisher_diag.data.ptr));
        const fisher_count = self.fisher_diag.data.len / @sizeOf(f32);
        var i: usize = 0;
        while (i < fisher_count) : (i += 1) {
            fisher_data[i] = 1.0;
        }
    }

    pub fn diagonalHessian(self: *SFD, loss_fn: LossFn, params: []*Tensor, context: ?*anyopaque) !Tensor {
        const hess = try Tensor.init(self.allocator, &.{params.len});
        const hess_data: [*]f32 = @ptrCast(@alignCast(hess.data.ptr));

        var i: usize = 0;
        while (i < params.len) : (i += 1) {
            const g = try self.gradient(loss_fn, params[i], context);
            defer g.deinit();

            const g_data: [*]const f32 = @ptrCast(@alignCast(g.data.ptr));
            const g_count = g.data.len / @sizeOf(f32);

            var fisher_approx: f32 = 0.0;
            var j: usize = 0;
            while (j < g_count) : (j += 1) {
                fisher_approx += g_data[j] * g_data[j];
            }

            hess_data[i] = if (g_count > 0) fisher_approx / @as(f32, @floatFromInt(g_count)) else 0.0;
        }

        return hess;
    }

    pub fn diagonalHessianSecondOrder(self: *SFD, loss_fn: LossFn, param: *Tensor, context: ?*anyopaque) !Tensor {
        const eps: f32 = 1e-4;
        const eps_sq = eps * eps;

        const hess = try Tensor.init(self.allocator, param.shape);
        const hess_data: [*]f32 = @ptrCast(@alignCast(hess.data.ptr));
        const param_data: [*]f32 = @ptrCast(@alignCast(param.data.ptr));
        const param_count = param.data.len / @sizeOf(f32);

        var i: usize = 0;
        while (i < param_count) : (i += 1) {
            const orig = param_data[i];

            param_data[i] = orig;
            const loss_center = try loss_fn(param, context);

            param_data[i] = orig + eps;
            const loss_plus = try loss_fn(param, context);

            param_data[i] = orig - eps;
            const loss_minus = try loss_fn(param, context);

            hess_data[i] = (loss_plus - 2.0 * loss_center + loss_minus) / eps_sq;

            param_data[i] = orig;
        }

        return hess;
    }

    pub fn gradient(self: *SFD, loss_fn: LossFn, param: *Tensor, context: ?*anyopaque) !Tensor {
        const eps: f32 = 1e-5;

        const grad = try Tensor.init(self.allocator, param.shape);
        const grad_data: [*]f32 = @ptrCast(@alignCast(grad.data.ptr));
        const param_data: [*]f32 = @ptrCast(@alignCast(param.data.ptr));
        const param_count = param.data.len / @sizeOf(f32);

        var i: usize = 0;
        while (i < param_count) : (i += 1) {
            const orig = param_data[i];

            param_data[i] = orig + eps;
            const loss_plus = try loss_fn(param, context);

            param_data[i] = orig - eps;
            const loss_minus = try loss_fn(param, context);

            grad_data[i] = (loss_plus - loss_minus) / (2.0 * eps);

            param_data[i] = orig;
        }

        return grad;
    }

    pub fn gradientForward(self: *SFD, loss_fn: LossFn, param: *Tensor, context: ?*anyopaque) !Tensor {
        const eps: f32 = 1e-5;

        const grad = try Tensor.init(self.allocator, param.shape);
        const grad_data: [*]f32 = @ptrCast(@alignCast(grad.data.ptr));
        const param_data: [*]f32 = @ptrCast(@alignCast(param.data.ptr));
        const param_count = param.data.len / @sizeOf(f32);

        const loss_center = try loss_fn(param, context);

        var i: usize = 0;
        while (i < param_count) : (i += 1) {
            const orig = param_data[i];

            param_data[i] = orig + eps;
            const loss_plus = try loss_fn(param, context);

            grad_data[i] = (loss_plus - loss_center) / eps;

            param_data[i] = orig;
        }

        return grad;
    }

    pub fn ampSchedule(self: *SFD, step: usize, warmup: usize, total: usize) f32 {
        _ = self;
        if (step < warmup) return @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup));
        const progress = @as(f32, @floatFromInt(step - warmup)) / @as(f32, @floatFromInt(total - warmup));
        return 0.5 * (1.0 + math.cos(math.pi * progress));
    }

    pub fn clipGradNorm(self: *SFD, grads: []*Tensor, max_norm: f32) !f32 {
        var total_norm: f32 = 0.0;
        var i: usize = 0;
        while (i < grads.len) : (i += 1) {
            const norm = grads[i].normL2();
            total_norm += norm * norm;
        }
        total_norm = math.sqrt(total_norm);
        if (total_norm > max_norm) {
            const scale = max_norm / (total_norm + self.eps);
            i = 0;
            while (i < grads.len) : (i += 1) {
                const g_data: [*]f32 = @ptrCast(@alignCast(grads[i].data.ptr));
                const g_count = grads[i].data.len / @sizeOf(f32);
                var j: usize = 0;
                while (j < g_count) : (j += 1) {
                    g_data[j] *= scale;
                }
            }
        }
        return total_norm;
    }

    pub fn saveState(self: *const SFD, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();
        try self.fisher_diag.save(writer);
        try self.momentum_buffer.save(writer);
        try self.velocity_buffer.save(writer);
        try writer.writeAll(mem.asBytes(&self.beta1));
        try writer.writeAll(mem.asBytes(&self.beta2));
        try writer.writeAll(mem.asBytes(&self.eps));
        try writer.writeAll(mem.asBytes(&self.clip_threshold));
        try writer.writeInt(usize, self.step_count, .Little);
    }

    pub fn loadState(self: *SFD, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();
        try self.fisher_diag.load(reader);
        try self.momentum_buffer.load(reader);
        try self.velocity_buffer.load(reader);
        var buf: [4]u8 = undefined;
        _ = try reader.readAll(&buf);
        self.beta1 = @bitCast(buf);
        _ = try reader.readAll(&buf);
        self.beta2 = @bitCast(buf);
        _ = try reader.readAll(&buf);
        self.eps = @bitCast(buf);
        _ = try reader.readAll(&buf);
        self.clip_threshold = @bitCast(buf);
        self.step_count = try reader.readInt(usize, .Little);
    }

    pub fn warmStart(self: *SFD, prev_diag: *const Tensor) !void {
        const fisher_data: [*]f32 = @ptrCast(@alignCast(self.fisher_diag.data.ptr));
        const prev_data: [*]const f32 = @ptrCast(@alignCast(prev_diag.data.ptr));
        const count = @min(self.fisher_diag.data.len, prev_diag.data.len) / @sizeOf(f32);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            fisher_data[i] = (fisher_data[i] + prev_data[i]) * 0.5;
        }
    }

    pub fn varianceReduction(self: *SFD, noise_grads: []const Tensor) !void {
        var avg_grad = try Tensor.zeros(self.allocator, self.fisher_diag.shape);
        defer avg_grad.deinit();
        const avg_data: [*]f32 = @ptrCast(@alignCast(avg_grad.data.ptr));
        const avg_count = avg_grad.data.len / @sizeOf(f32);
        var i: usize = 0;
        while (i < noise_grads.len) : (i += 1) {
            const ng = noise_grads[i];
            const ng_data: [*]const f32 = @ptrCast(@alignCast(ng.data.ptr));
            const ng_count = ng.data.len / @sizeOf(f32);
            var j: usize = 0;
            while (j < @min(avg_count, ng_count)) : (j += 1) {
                avg_data[j] += ng_data[j] * ng_data[j];
            }
        }
        i = 0;
        while (i < avg_count) : (i += 1) {
            avg_data[i] /= @as(f32, @floatFromInt(noise_grads.len));
        }
        const fisher_data: [*]f32 = @ptrCast(@alignCast(self.fisher_diag.data.ptr));
        const fisher_count = self.fisher_diag.data.len / @sizeOf(f32);
        i = 0;
        while (i < @min(fisher_count, avg_count)) : (i += 1) {
            fisher_data[i] = math.max(0.0, fisher_data[i] - avg_data[i]);
        }
    }
};

test "SFD update" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();
    var grads = try Tensor.init(gpa, &.{4});
    defer grads.deinit();
    const grad_data: [*]f32 = @ptrCast(@alignCast(grads.data.ptr));
    grad_data[0] = 1.0;
    grad_data[1] = 2.0;
    grad_data[2] = 3.0;
    grad_data[3] = 4.0;
    var params = try Tensor.init(gpa, &.{4});
    defer params.deinit();
    @memset(params.data, 0);
    try sfd.update(&grads, &params, 0.1);
    const param_data: [*]const f32 = @ptrCast(@alignCast(params.data.ptr));
    try testing.expect(param_data[0] < 0);
}

test "SFD spectral clip" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();
    var t = try Tensor.init(gpa, &.{ 2, 2 });
    defer t.deinit();
    const t_data: [*]f32 = @ptrCast(@alignCast(t.data.ptr));
    t_data[0] = 2.0;
    t_data[1] = 0.0;
    t_data[2] = 0.0;
    t_data[3] = 2.0;
    try sfd.spectralClip(&t, 1.0);
    try testing.expectApproxEqAbs(t_data[0], 1.0, 1e-5);
}

test "SFD adaptive LR" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 1);
    defer sfd.deinit();
    const lr = sfd.adaptiveLR(1.0, 1.0);
    try testing.expectApproxEqAbs(lr, 1.0, 1e-5);
}

test "SFD clip grad norm" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();
    var g1 = try Tensor.init(gpa, &.{2});
    defer g1.deinit();
    const g1_data: [*]f32 = @ptrCast(@alignCast(g1.data.ptr));
    g1_data[0] = 10.0;
    g1_data[1] = 10.0;
    var g2 = try Tensor.init(gpa, &.{2});
    defer g2.deinit();
    @memset(g2.data, 0);
    const norm = try sfd.clipGradNorm(&.{ &g1, &g2 }, 5.0);
    try testing.expect(norm <= 15.0);
}

test "SFD AMP schedule" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 1);
    defer sfd.deinit();
    const sched = sfd.ampSchedule(500, 100, 1000);
    try testing.expect(sched > 0.0 and sched < 1.0);
}

test "SFD save load" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 3);
    defer sfd.deinit();
    const fisher_data: [*]f32 = @ptrCast(@alignCast(sfd.fisher_diag.data.ptr));
    fisher_data[0] = 1.1;
    fisher_data[1] = 1.2;
    fisher_data[2] = 1.3;
    try sfd.saveState("test_sfd.bin");
    defer {
        std.fs.cwd().deleteFile("test_sfd.bin") catch |err| {
            std.log.warn("Failed to delete test file: {}", .{err});
        };
    }
    var sfd2 = try SFD.init(gpa, 3);
    defer sfd2.deinit();
    try sfd2.loadState("test_sfd.bin");
    const sfd2_data: [*]const f32 = @ptrCast(@alignCast(sfd2.fisher_diag.data.ptr));
    try testing.expectApproxEqAbs(sfd2_data[0], 1.1, 1e-5);
}

test "SFD accumulate Fisher" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();
    var g1 = try Tensor.init(gpa, &.{2});
    defer g1.deinit();
    const g1_data: [*]f32 = @ptrCast(@alignCast(g1.data.ptr));
    g1_data[0] = 1.0;
    g1_data[1] = 2.0;
    var g2 = try Tensor.init(gpa, &.{2});
    defer g2.deinit();
    const g2_data: [*]f32 = @ptrCast(@alignCast(g2.data.ptr));
    g2_data[0] = 3.0;
    g2_data[1] = 4.0;
    try sfd.accumulateFisher(&.{ g1, g2 });
    const fisher_data: [*]const f32 = @ptrCast(@alignCast(sfd.fisher_diag.data.ptr));
    try testing.expect(fisher_data[0] > 1.0);
}

test "SFD reset Fisher" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 3);
    defer sfd.deinit();
    const fisher_data: [*]f32 = @ptrCast(@alignCast(sfd.fisher_diag.data.ptr));
    fisher_data[0] = 5.0;
    fisher_data[1] = 10.0;
    fisher_data[2] = 15.0;
    sfd.resetFisher();
    try testing.expectEqual(@as(f32, 1.0), fisher_data[0]);
    try testing.expectEqual(@as(f32, 1.0), fisher_data[1]);
    try testing.expectEqual(@as(f32, 1.0), fisher_data[2]);
}

test "SFD warm start" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();
    var prev = try Tensor.init(gpa, &.{2});
    defer prev.deinit();
    const prev_data: [*]f32 = @ptrCast(@alignCast(prev.data.ptr));
    prev_data[0] = 2.0;
    prev_data[1] = 4.0;
    try sfd.warmStart(&prev);
    const fisher_data: [*]const f32 = @ptrCast(@alignCast(sfd.fisher_diag.data.ptr));
    try testing.expectApproxEqAbs(fisher_data[0], 1.5, 1e-5);
    try testing.expectApproxEqAbs(fisher_data[1], 2.5, 1e-5);
}

test "SFD variance reduction" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();
    var g1 = try Tensor.init(gpa, &.{2});
    defer g1.deinit();
    const g1_data: [*]f32 = @ptrCast(@alignCast(g1.data.ptr));
    g1_data[0] = 0.1;
    g1_data[1] = 0.2;
    var g2 = try Tensor.init(gpa, &.{2});
    defer g2.deinit();
    const g2_data: [*]f32 = @ptrCast(@alignCast(g2.data.ptr));
    g2_data[0] = 0.3;
    g2_data[1] = 0.4;
    try sfd.varianceReduction(&.{ g1, g2 });
    const fisher_data: [*]const f32 = @ptrCast(@alignCast(sfd.fisher_diag.data.ptr));
    try testing.expect(fisher_data[0] >= 0.0);
}

fn testQuadraticLoss(params: *const Tensor, context: ?*anyopaque) !f32 {
    _ = context;
    const param_data: [*]const f32 = @ptrCast(@alignCast(params.data.ptr));
    const param_count = params.data.len / @sizeOf(f32);

    var loss: f32 = 0.0;
    var i: usize = 0;
    while (i < param_count) : (i += 1) {
        loss += param_data[i] * param_data[i];
    }
    return loss;
}

test "SFD gradient computation" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 3);
    defer sfd.deinit();

    var params = try Tensor.init(gpa, &.{3});
    defer params.deinit();
    const param_data: [*]f32 = @ptrCast(@alignCast(params.data.ptr));
    param_data[0] = 1.0;
    param_data[1] = 2.0;
    param_data[2] = 3.0;

    var grad = try sfd.gradient(testQuadraticLoss, &params, null);
    defer grad.deinit();

    const grad_data: [*]const f32 = @ptrCast(@alignCast(grad.data.ptr));

    try testing.expectApproxEqAbs(grad_data[0], 2.0, 1e-3);
    try testing.expectApproxEqAbs(grad_data[1], 4.0, 1e-3);
    try testing.expectApproxEqAbs(grad_data[2], 6.0, 1e-3);
}

test "SFD diagonal Hessian" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 2);
    defer sfd.deinit();

    var p1 = try Tensor.init(gpa, &.{2});
    defer p1.deinit();
    const p1_data: [*]f32 = @ptrCast(@alignCast(p1.data.ptr));
    p1_data[0] = 1.0;
    p1_data[1] = 2.0;

    var p2 = try Tensor.init(gpa, &.{2});
    defer p2.deinit();
    const p2_data: [*]f32 = @ptrCast(@alignCast(p2.data.ptr));
    p2_data[0] = 3.0;
    p2_data[1] = 4.0;

    var hess = try sfd.diagonalHessian(testQuadraticLoss, &.{ &p1, &p2 }, null);
    defer hess.deinit();

    const hess_data: [*]const f32 = @ptrCast(@alignCast(hess.data.ptr));
    try testing.expect(hess_data[0] > 0.0);
    try testing.expect(hess_data[1] > 0.0);
}

test "SFD second-order Hessian" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 3);
    defer sfd.deinit();

    var params = try Tensor.init(gpa, &.{3});
    defer params.deinit();
    const param_data: [*]f32 = @ptrCast(@alignCast(params.data.ptr));
    param_data[0] = 1.0;
    param_data[1] = 2.0;
    param_data[2] = 3.0;

    var hess = try sfd.diagonalHessianSecondOrder(testQuadraticLoss, &params, null);
    defer hess.deinit();

    const hess_data: [*]const f32 = @ptrCast(@alignCast(hess.data.ptr));

    try testing.expectApproxEqAbs(hess_data[0], 2.0, 1e-2);
    try testing.expectApproxEqAbs(hess_data[1], 2.0, 1e-2);
    try testing.expectApproxEqAbs(hess_data[2], 2.0, 1e-2);
}

pub const SFDOptimizer = SFD;
