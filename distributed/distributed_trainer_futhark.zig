const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const RSFAccelerator = @import("../hw/accel/accel_interface.zig").RSFAccelerator;
const FutharkArray2DF16 = @import("../hw/accel/accel_interface.zig").FutharkArray2DF16;
const PinnedMemory = @import("../hw/accel/accel_interface.zig").PinnedMemory;

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: RSFAccelerator,
    model_dim: usize,
    local_batch_size: usize,
    global_step: usize,
    learning_rate: f16,
    momentum: f16,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        const vocab = &[_][]const u8{
            "a", "about", "all", "also", "and", "as", "at", "be", "because", "but", "by", "can", "come", "could", "day", "do", "even", "find", "first", "for", "from", "get", "give", "go", "have", "he", "her", "here", "him", "his", "how", "I", "if", "in", "into", "it", "its", "just", "know", "like", "look", "make", "man", "many", "me", "more", "my", "new", "no", "not", "now", "of", "on", "one", "only", "or", "other", "our", "out", "people", "say", "see", "she", "so", "some", "take", "tell", "than", "that", "the", "their", "them", "then", "there", "these", "they", "thing", "think", "this", "those", "time", "to", "two", "up", "use", "very", "want", "way", "we", "well", "what", "when", "which", "who", "will", "with", "would", "year", "you", "your"
        };
        const empty_anchors: []const []const u8 = &.{};
        var tokenizer = try MGT.init(allocator, vocab, empty_anchors);
        errdefer tokenizer.deinit();

        var accelerator = try RSFAccelerator.init(model_dim);
        errdefer accelerator.deinit();

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer,
            .accelerator = accelerator,
            .model_dim = model_dim,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = @floatCast(0.001),
            .momentum = @floatCast(0.9),
        };
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        var line_count: usize = 0;

        {
            const count_file = try std.fs.openFileAbsolute(dataset_path, .{});
            defer count_file.close();

            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();

            while (true) : (line_count += 1) {
                count_stream.skipUntilDelimiterOrEof('\n') catch |err| switch (err) {
                    error.EndOfStream => break,
                    else => return err,
                };
            }
        }

        if (line_count == 0) {
            std.debug.print("[Rank {d}] ERROR: Dataset is empty\n", .{self.coordinator.rank});
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

        var current_line: usize = 0;
        while (current_line < start_line) : (current_line += 1) {
            load_stream.skipUntilDelimiterOrEof('\n') catch |err| switch (err) {
                error.EndOfStream => return error.UnexpectedEndOfFile,
                else => return err,
            };
        }

        while (current_line < end_line) : (current_line += 1) {
            const line = load_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', 1024 * 1024 * 10) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            } orelse break;
            defer self.allocator.free(line);

            if (line.len == 0) continue;

            const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, line, .{}) catch continue;
            defer parsed.deinit();

            if (parsed.value.object.get("text")) |text_value| {
                if (text_value.jsonType != .string) continue;
                const text = text_value.string;
                const text_copy = try self.allocator.dupe(u8, text);
                try samples.append(text_copy);
            }
        }

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples (lines {d}-{d} of {d} total)\n",
                .{self.coordinator.rank, samples.items.len, start_line, end_line - 1, line_count});
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) {
            const batch_end = @min(batch_start + self.local_batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            const loss = try self.trainStepFuthark(batch);
            total_loss += loss;
            num_batches += 1;

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{self.global_step, loss});
            }

            self.global_step += 1;
            batch_start = batch_end;
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        const loss_and_count_dev = try self.coordinator.allocDeviceMemory(2 * @sizeOf(f32));
        defer self.coordinator.freeDeviceMemory(loss_and_count_dev);

        try self.coordinator.copyHostToDevice(loss_and_count_dev, &loss_and_count, 2 * @sizeOf(f32));
        try self.coordinator.allReduceFloat32(loss_and_count_dev, loss_and_count_dev, 2);
        try self.coordinator.copyDeviceToHost(&loss_and_count, loss_and_count_dev, 2 * @sizeOf(f32));
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

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !f32 {
        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| {
                list.deinit();
            }
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            try self.tokenizer.encode(text, &token_list);
            try token_lists.append(token_list);
        }

        const max_seq_len = blk: {
            var max_len: usize = 0;
            for (token_lists.items) |list| {
                max_len = @max(max_len, list.items.len);
            }
            break :blk max_len;
        };

        if (max_seq_len == 0) return 0.0;

        const data_size = batch.len * max_seq_len * self.model_dim * @sizeOf(f16);
        var pinned_mem = try PinnedMemory.alloc(data_size);
        defer pinned_mem.free();

        var input_f16_data = pinned_mem.asSlice(f16);
        @memset(input_f16_data, 0);

        var batch_idx: usize = 0;
        while (batch_idx < token_lists.items.len) : (batch_idx += 1) {
            const list = token_lists.items[batch_idx].items;
            var seq_idx: usize = 0;
            while (seq_idx < list.len) : (seq_idx += 1) {
                const token = list[seq_idx];
                if (token >= self.model_dim) return error.InvalidToken;
                const base_idx = (batch_idx * max_seq_len + seq_idx) * self.model_dim;
                const token_idx = token;
                input_f16_data[base_idx + token_idx] = @as(f16, 1.0);
            }
        }

        const total_rows = batch.len * max_seq_len;
        const inputs = try FutharkArray2DF16.newFromFlat(self.accelerator.ctx, input_f16_data, total_rows, self.model_dim);
        defer inputs.free(self.accelerator.ctx);

        const targets = try FutharkArray2DF16.newFromFlat(self.accelerator.ctx, input_f16_data, total_rows, self.model_dim);
        defer targets.free(self.accelerator.ctx);

        const loss_f16 = try self.accelerator.trainingStep(
            inputs,
            targets,
            self.learning_rate,
            self.momentum,
        );

        const weights_s_ptr = try self.accelerator.getWeightsSDevicePointer();
        const weights_t_ptr = try self.accelerator.getWeightsTDevicePointer();

        try self.coordinator.allReduceFloat16(weights_s_ptr, weights_s_ptr, self.model_dim * self.model_dim);
        try self.coordinator.allReduceFloat16(weights_t_ptr, weights_t_ptr, self.model_dim * self.model_dim);
        try self.coordinator.synchronize();

        const world_size_f16: f16 = @floatFromInt(self.coordinator.world_size);
        const scale_factor: f16 = @as(f16, 1.0) / world_size_f16;
        try self.accelerator.scaleWeightsInplace(scale_factor);
        try self.accelerator.sync();

        return @floatFromInt(loss_f16);
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            return;
        }

        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        var writer = file.writer();

        try writer.writeInt(usize, self.global_step, .Little);
        try writer.writeInt(usize, self.model_dim, .Little);

        const weights_s_vals = try self.accelerator.weights_s.values(self.accelerator.ctx, self.allocator);
        defer {
            for (weights_s_vals) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(weights_s_vals);
        }

        for (weights_s_vals) |row| {
            for (row) |weight| {
                try writer.writeAll(std.mem.asBytes(&weight));
            }
        }

        const weights_t_vals = try self.accelerator.weights_t.values(self.accelerator.ctx, self.allocator);
        defer {
            for (weights_t_vals) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(weights_t_vals);
        }

        for (weights_t_vals) |row| {
            for (row) |weight| {
                try writer.writeAll(std.mem.asBytes(&weight));
            }
        }

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{path, self.global_step});
    }
};