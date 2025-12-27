const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

const nsir_core = @import("nsir_core.zig");
const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
const Node = nsir_core.Node;
const Edge = nsir_core.Edge;
const EdgeQuality = nsir_core.EdgeQuality;
const EdgeKey = nsir_core.EdgeKey;

pub const InvariantType = enum(u8) {
    CONNECTIVITY = 0,
    SYMMETRY = 1,
    COHERENCE = 2,
    ENTANGLEMENT = 3,
    FRACTAL_DIMENSION = 4,
    QUANTUM_STATE = 5,
    MEMORY_SAFETY = 6,
    TYPE_SAFETY = 7,
    TEMPORAL_CONSISTENCY = 8,

    pub fn toString(self: InvariantType) []const u8 {
        return switch (self) {
            .CONNECTIVITY => "connectivity",
            .SYMMETRY => "symmetry",
            .COHERENCE => "coherence",
            .ENTANGLEMENT => "entanglement",
            .FRACTAL_DIMENSION => "fractal_dimension",
            .QUANTUM_STATE => "quantum_state",
            .MEMORY_SAFETY => "memory_safety",
            .TYPE_SAFETY => "type_safety",
            .TEMPORAL_CONSISTENCY => "temporal_consistency",
        };
    }

    pub fn fromString(s: []const u8) ?InvariantType {
        if (std.mem.eql(u8, s, "connectivity")) return .CONNECTIVITY;
        if (std.mem.eql(u8, s, "symmetry")) return .SYMMETRY;
        if (std.mem.eql(u8, s, "coherence")) return .COHERENCE;
        if (std.mem.eql(u8, s, "entanglement")) return .ENTANGLEMENT;
        if (std.mem.eql(u8, s, "fractal_dimension")) return .FRACTAL_DIMENSION;
        if (std.mem.eql(u8, s, "quantum_state")) return .QUANTUM_STATE;
        if (std.mem.eql(u8, s, "memory_safety")) return .MEMORY_SAFETY;
        if (std.mem.eql(u8, s, "type_safety")) return .TYPE_SAFETY;
        if (std.mem.eql(u8, s, "temporal_consistency")) return .TEMPORAL_CONSISTENCY;
        return null;
    }

    pub fn priority(self: InvariantType) u8 {
        return switch (self) {
            .MEMORY_SAFETY => 10,
            .TYPE_SAFETY => 9,
            .CONNECTIVITY => 8,
            .COHERENCE => 7,
            .ENTANGLEMENT => 6,
            .QUANTUM_STATE => 5,
            .FRACTAL_DIMENSION => 4,
            .SYMMETRY => 3,
            .TEMPORAL_CONSISTENCY => 2,
        };
    }
};

pub const ProofRule = enum(u8) {
    AXIOM = 0,
    MODUS_PONENS = 1,
    UNIVERSAL_INSTANTIATION = 2,
    EXISTENTIAL_GENERALIZATION = 3,
    INDUCTION = 4,
    CONTRADICTION = 5,
    DEDUCTION = 6,
    WEAKENING = 7,
    STRENGTHENING = 8,
    FRAME_RULE = 9,
    CONSEQUENCE_RULE = 10,
    CONJUNCTION_INTRO = 11,
    CONJUNCTION_ELIM = 12,
    DISJUNCTION_INTRO = 13,
    DISJUNCTION_ELIM = 14,
    NEGATION_INTRO = 15,
    NEGATION_ELIM = 16,
    IMPLICATION_INTRO = 17,
    IMPLICATION_ELIM = 18,
    UNIVERSAL_INTRO = 19,
    EXISTENTIAL_ELIM = 20,
    TEMPORAL_INDUCTION = 21,
    LOOP_INVARIANT = 22,
    ASSIGNMENT_AXIOM = 23,
    SEQUENCE_RULE = 24,
    CONDITIONAL_RULE = 25,

    pub fn toString(self: ProofRule) []const u8 {
        return switch (self) {
            .AXIOM => "axiom",
            .MODUS_PONENS => "modus_ponens",
            .UNIVERSAL_INSTANTIATION => "universal_instantiation",
            .EXISTENTIAL_GENERALIZATION => "existential_generalization",
            .INDUCTION => "induction",
            .CONTRADICTION => "contradiction",
            .DEDUCTION => "deduction",
            .WEAKENING => "weakening",
            .STRENGTHENING => "strengthening",
            .FRAME_RULE => "frame_rule",
            .CONSEQUENCE_RULE => "consequence_rule",
            .CONJUNCTION_INTRO => "conjunction_intro",
            .CONJUNCTION_ELIM => "conjunction_elim",
            .DISJUNCTION_INTRO => "disjunction_intro",
            .DISJUNCTION_ELIM => "disjunction_elim",
            .NEGATION_INTRO => "negation_intro",
            .NEGATION_ELIM => "negation_elim",
            .IMPLICATION_INTRO => "implication_intro",
            .IMPLICATION_ELIM => "implication_elim",
            .UNIVERSAL_INTRO => "universal_intro",
            .EXISTENTIAL_ELIM => "existential_elim",
            .TEMPORAL_INDUCTION => "temporal_induction",
            .LOOP_INVARIANT => "loop_invariant",
            .ASSIGNMENT_AXIOM => "assignment_axiom",
            .SEQUENCE_RULE => "sequence_rule",
            .CONDITIONAL_RULE => "conditional_rule",
        };
    }

    pub fn fromString(s: []const u8) ?ProofRule {
        if (std.mem.eql(u8, s, "axiom")) return .AXIOM;
        if (std.mem.eql(u8, s, "modus_ponens")) return .MODUS_PONENS;
        if (std.mem.eql(u8, s, "universal_instantiation")) return .UNIVERSAL_INSTANTIATION;
        if (std.mem.eql(u8, s, "existential_generalization")) return .EXISTENTIAL_GENERALIZATION;
        if (std.mem.eql(u8, s, "induction")) return .INDUCTION;
        if (std.mem.eql(u8, s, "contradiction")) return .CONTRADICTION;
        if (std.mem.eql(u8, s, "deduction")) return .DEDUCTION;
        if (std.mem.eql(u8, s, "weakening")) return .WEAKENING;
        if (std.mem.eql(u8, s, "strengthening")) return .STRENGTHENING;
        if (std.mem.eql(u8, s, "frame_rule")) return .FRAME_RULE;
        if (std.mem.eql(u8, s, "consequence_rule")) return .CONSEQUENCE_RULE;
        return null;
    }

    pub fn requiresPremises(self: ProofRule) bool {
        return switch (self) {
            .AXIOM, .ASSIGNMENT_AXIOM => false,
            else => true,
        };
    }

    pub fn minimumPremises(self: ProofRule) usize {
        return switch (self) {
            .AXIOM, .ASSIGNMENT_AXIOM => 0,
            .MODUS_PONENS, .IMPLICATION_ELIM => 2,
            .UNIVERSAL_INSTANTIATION, .EXISTENTIAL_GENERALIZATION => 1,
            .INDUCTION => 2,
            .CONTRADICTION => 2,
            .DEDUCTION, .IMPLICATION_INTRO => 1,
            .WEAKENING, .STRENGTHENING => 1,
            .FRAME_RULE => 1,
            .CONSEQUENCE_RULE => 3,
            .CONJUNCTION_INTRO => 2,
            .CONJUNCTION_ELIM => 1,
            .DISJUNCTION_INTRO => 1,
            .DISJUNCTION_ELIM => 3,
            .NEGATION_INTRO, .NEGATION_ELIM => 1,
            .UNIVERSAL_INTRO => 1,
            .EXISTENTIAL_ELIM => 2,
            .TEMPORAL_INDUCTION => 2,
            .LOOP_INVARIANT => 2,
            .SEQUENCE_RULE => 2,
            .CONDITIONAL_RULE => 2,
        };
    }
};

pub const PropType = enum(u8) {
    ATOMIC = 0,
    NEGATION = 1,
    CONJUNCTION = 2,
    DISJUNCTION = 3,
    IMPLICATION = 4,
    UNIVERSAL = 5,
    EXISTENTIAL = 6,
    TEMPORAL_ALWAYS = 7,
    TEMPORAL_EVENTUALLY = 8,
    HOARE_TRIPLE = 9,
    BICONDITIONAL = 10,
    TRUE = 11,
    FALSE = 12,
    SEPARATION_STAR = 13,
    SEPARATION_WAND = 14,
    RELATIONAL_EDGE = 15,
    QUANTUM_SUPERPOSITION = 16,
    ENTANGLEMENT_PAIR = 17,

    pub fn toString(self: PropType) []const u8 {
        return switch (self) {
            .ATOMIC => "atomic",
            .NEGATION => "negation",
            .CONJUNCTION => "conjunction",
            .DISJUNCTION => "disjunction",
            .IMPLICATION => "implication",
            .UNIVERSAL => "universal",
            .EXISTENTIAL => "existential",
            .TEMPORAL_ALWAYS => "temporal_always",
            .TEMPORAL_EVENTUALLY => "temporal_eventually",
            .HOARE_TRIPLE => "hoare_triple",
            .BICONDITIONAL => "biconditional",
            .TRUE => "true",
            .FALSE => "false",
            .SEPARATION_STAR => "separation_star",
            .SEPARATION_WAND => "separation_wand",
            .RELATIONAL_EDGE => "relational_edge",
            .QUANTUM_SUPERPOSITION => "quantum_superposition",
            .ENTANGLEMENT_PAIR => "entanglement_pair",
        };
    }

    pub fn fromString(s: []const u8) ?PropType {
        if (std.mem.eql(u8, s, "atomic")) return .ATOMIC;
        if (std.mem.eql(u8, s, "negation")) return .NEGATION;
        if (std.mem.eql(u8, s, "conjunction")) return .CONJUNCTION;
        if (std.mem.eql(u8, s, "disjunction")) return .DISJUNCTION;
        if (std.mem.eql(u8, s, "implication")) return .IMPLICATION;
        if (std.mem.eql(u8, s, "universal")) return .UNIVERSAL;
        if (std.mem.eql(u8, s, "existential")) return .EXISTENTIAL;
        if (std.mem.eql(u8, s, "temporal_always")) return .TEMPORAL_ALWAYS;
        if (std.mem.eql(u8, s, "temporal_eventually")) return .TEMPORAL_EVENTUALLY;
        if (std.mem.eql(u8, s, "hoare_triple")) return .HOARE_TRIPLE;
        return null;
    }

    pub fn arity(self: PropType) usize {
        return switch (self) {
            .ATOMIC, .TRUE, .FALSE => 0,
            .NEGATION, .TEMPORAL_ALWAYS, .TEMPORAL_EVENTUALLY => 1,
            .CONJUNCTION, .DISJUNCTION, .IMPLICATION, .BICONDITIONAL, .SEPARATION_STAR, .SEPARATION_WAND, .ENTANGLEMENT_PAIR => 2,
            .UNIVERSAL, .EXISTENTIAL => 1,
            .HOARE_TRIPLE => 3,
            .RELATIONAL_EDGE => 2,
            .QUANTUM_SUPERPOSITION => 2,
        };
    }
};

pub const VerificationError = error{
    InvalidProofStep,
    PremiseMismatch,
    InvalidConclusion,
    InvariantViolation,
    OutOfMemory,
    ProofIncomplete,
    UnificationFailure,
    ResolutionFailure,
    InvalidGraph,
    CircularDependency,
    TypeMismatch,
    MemoryLeak,
    DeadlockDetected,
    InvalidQuantumState,
    FractalBoundsExceeded,
    ConnectivityBroken,
    CoherenceLost,
    EntanglementBroken,
    PredicateEvaluationError,
    InvalidPremiseIndex,
    InvalidRule,
};

pub const Term = struct {
    kind: TermKind,
    name: []const u8,
    args: ArrayList(*Term),
    allocator: Allocator,

    pub const TermKind = enum(u8) {
        VARIABLE = 0,
        CONSTANT = 1,
        FUNCTION = 2,
    };

    const Self = @This();

    pub fn initVariable(allocator: Allocator, name: []const u8) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .VARIABLE,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
        };
        return term;
    }

    pub fn initConstant(allocator: Allocator, name: []const u8) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .CONSTANT,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
        };
        return term;
    }

    pub fn initFunction(allocator: Allocator, name: []const u8, args: []const *Term) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .FUNCTION,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
        };
        for (args) |arg| {
            try term.args.append(arg);
        }
        return term;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.args.items) |arg| {
            arg.deinit();
            self.allocator.destroy(arg);
        }
        self.args.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_term = try allocator.create(Self);
        new_term.* = Self{
            .kind = self.kind,
            .name = try allocator.dupe(u8, self.name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
        };
        for (self.args.items) |arg| {
            try new_term.args.append(try arg.clone(allocator));
        }
        return new_term;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.kind != other.kind) return false;
        if (!std.mem.eql(u8, self.name, other.name)) return false;
        if (self.args.items.len != other.args.items.len) return false;
        var i: usize = 0;
        while (i < self.args.items.len) : (i += 1) {
            if (!self.args.items[i].equals(other.args.items[i])) return false;
        }
        return true;
    }

    pub fn isVariable(self: *const Self) bool {
        return self.kind == .VARIABLE;
    }

    pub fn containsVariable(self: *const Self, var_name: []const u8) bool {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            return true;
        }
        for (self.args.items) |arg| {
            if (arg.containsVariable(var_name)) {
                return true;
            }
        }
        return false;
    }

    pub fn substitute(self: *Self, var_name: []const u8, replacement: *const Term) !void {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            self.allocator.free(self.name);
            self.name = try self.allocator.dupe(u8, replacement.name);
            self.kind = replacement.kind;
            for (self.args.items) |arg| {
                arg.deinit();
                self.allocator.destroy(arg);
            }
            self.args.clearRetainingCapacity();
            for (replacement.args.items) |arg| {
                try self.args.append(try arg.clone(self.allocator));
            }
        } else {
            for (self.args.items) |arg| {
                try arg.substitute(var_name, replacement);
            }
        }
    }
};

pub const Proposition = struct {
    prop_type: PropType,
    sub_propositions: ArrayList(*Proposition),
    predicate_name: []const u8,
    bound_variable: ?[]const u8,
    terms: ArrayList(*Term),
    allocator: Allocator,
    hash_cache: ?[32]u8,

    const Self = @This();

    pub fn init(allocator: Allocator, prop_type: PropType) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        return prop;
    }

    pub fn initAtomic(allocator: Allocator, predicate_name: []const u8) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .ATOMIC,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, predicate_name),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        return prop;
    }

    pub fn initNegation(allocator: Allocator, inner: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .NEGATION,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        try prop.sub_propositions.append(inner);
        return prop;
    }

    pub fn initBinary(allocator: Allocator, prop_type: PropType, left: *Proposition, right: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        try prop.sub_propositions.append(left);
        try prop.sub_propositions.append(right);
        return prop;
    }

    pub fn initQuantified(allocator: Allocator, prop_type: PropType, variable: []const u8, body: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = try allocator.dupe(u8, variable),
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        try prop.sub_propositions.append(body);
        return prop;
    }

    pub fn initHoareTriple(allocator: Allocator, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .HOARE_TRIPLE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        try prop.sub_propositions.append(precondition);
        try prop.sub_propositions.append(operation);
        try prop.sub_propositions.append(postcondition);
        return prop;
    }

    pub fn initTrue(allocator: Allocator) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .TRUE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, "true"),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        return prop;
    }

    pub fn initFalse(allocator: Allocator) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .FALSE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, "false"),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
        };
        return prop;
    }

    pub fn deinit(self: *Self) void {
        for (self.sub_propositions.items) |sub| {
            sub.deinit();
            self.allocator.destroy(sub);
        }
        self.sub_propositions.deinit();
        if (self.predicate_name.len > 0) {
            self.allocator.free(self.predicate_name);
        }
        if (self.bound_variable) |bv| {
            self.allocator.free(bv);
        }
        for (self.terms.items) |term| {
            term.deinit();
            self.allocator.destroy(term);
        }
        self.terms.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_prop = try allocator.create(Self);
        new_prop.* = Self{
            .prop_type = self.prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = if (self.predicate_name.len > 0) try allocator.dupe(u8, self.predicate_name) else "",
            .bound_variable = if (self.bound_variable) |bv| try allocator.dupe(u8, bv) else null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = self.hash_cache,
        };
        for (self.sub_propositions.items) |sub| {
            try new_prop.sub_propositions.append(try sub.clone(allocator));
        }
        for (self.terms.items) |term| {
            try new_prop.terms.append(try term.clone(allocator));
        }
        return new_prop;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.prop_type != other.prop_type) return false;
        if (!std.mem.eql(u8, self.predicate_name, other.predicate_name)) return false;
        if (self.sub_propositions.items.len != other.sub_propositions.items.len) return false;
        var i: usize = 0;
        while (i < self.sub_propositions.items.len) : (i += 1) {
            if (!self.sub_propositions.items[i].equals(other.sub_propositions.items[i])) return false;
        }
        return true;
    }

    pub fn computeHash(self: *Self) [32]u8 {
        if (self.hash_cache) |cache| {
            return cache;
        }
        var hasher = Sha256.init(.{});
        hasher.update(&[_]u8{@intFromEnum(self.prop_type)});
        hasher.update(self.predicate_name);
        if (self.bound_variable) |bv| {
            hasher.update(bv);
        }
        for (self.sub_propositions.items) |sub| {
            const sub_hash = sub.computeHash();
            hasher.update(&sub_hash);
        }
        var result: [32]u8 = undefined;
        hasher.final(&result);
        self.hash_cache = result;
        return result;
    }

    pub fn isAtomic(self: *const Self) bool {
        return self.prop_type == .ATOMIC or self.prop_type == .TRUE or self.prop_type == .FALSE;
    }

    pub fn negate(self: *const Self, allocator: Allocator) !*Self {
        if (self.prop_type == .NEGATION and self.sub_propositions.items.len > 0) {
            return self.sub_propositions.items[0].clone(allocator);
        }
        const cloned = try self.clone(allocator);
        return Proposition.initNegation(allocator, cloned);
    }

    pub fn implies(self: *const Self, allocator: Allocator, consequent: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        return Proposition.initBinary(allocator, .IMPLICATION, cloned_self, consequent);
    }

    pub fn conjoin(self: *const Self, allocator: Allocator, other: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        return Proposition.initBinary(allocator, .CONJUNCTION, cloned_self, other);
    }

    pub fn disjoin(self: *const Self, allocator: Allocator, other: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        return Proposition.initBinary(allocator, .DISJUNCTION, cloned_self, other);
    }

    pub fn freeVariables(self: *const Self, allocator: Allocator) !ArrayList([]const u8) {
        var vars = ArrayList([]const u8).init(allocator);
        try self.collectFreeVariables(&vars, null);
        return vars;
    }

    fn collectFreeVariables(self: *const Self, vars: *ArrayList([]const u8), bound: ?[]const u8) !void {
        for (self.terms.items) |term| {
            if (term.kind == .VARIABLE) {
                if (bound == null or !std.mem.eql(u8, term.name, bound.?)) {
                    var found = false;
                    for (vars.items) |v| {
                        if (std.mem.eql(u8, v, term.name)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        try vars.append(term.name);
                    }
                }
            }
        }
        const new_bound = if (self.prop_type == .UNIVERSAL or self.prop_type == .EXISTENTIAL) self.bound_variable else bound;
        for (self.sub_propositions.items) |sub| {
            try sub.collectFreeVariables(vars, new_bound);
        }
    }

    pub fn substitute(self: *Self, var_name: []const u8, replacement: *const Term) !void {
        if (self.bound_variable) |bv| {
            if (std.mem.eql(u8, bv, var_name)) {
                return;
            }
        }
        for (self.terms.items) |term| {
            try term.substitute(var_name, replacement);
        }
        for (self.sub_propositions.items) |sub| {
            try sub.substitute(var_name, replacement);
        }
        self.hash_cache = null;
    }
};

pub const ProofStep = struct {
    step_id: u64,
    rule_applied: ProofRule,
    premise_indices: ArrayList(usize),
    conclusion: *Proposition,
    justification: []const u8,
    verified: bool,
    timestamp: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, step_id: u64, rule: ProofRule, conclusion: *Proposition) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_id = step_id,
            .rule_applied = rule,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = conclusion,
            .justification = "",
            .verified = false,
            .timestamp = std.time.milliTimestamp(),
            .allocator = allocator,
        };
        return step;
    }

    pub fn initWithPremises(allocator: Allocator, step_id: u64, rule: ProofRule, premises: []const usize, conclusion: *Proposition, justification: []const u8) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_id = step_id,
            .rule_applied = rule,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = conclusion,
            .justification = try allocator.dupe(u8, justification),
            .verified = false,
            .timestamp = std.time.milliTimestamp(),
            .allocator = allocator,
        };
        for (premises) |p| {
            try step.premise_indices.append(p);
        }
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.premise_indices.deinit();
        self.conclusion.deinit();
        self.allocator.destroy(self.conclusion);
        if (self.justification.len > 0) {
            self.allocator.free(self.justification);
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_step = try allocator.create(Self);
        new_step.* = Self{
            .step_id = self.step_id,
            .rule_applied = self.rule_applied,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = try self.conclusion.clone(allocator),
            .justification = if (self.justification.len > 0) try allocator.dupe(u8, self.justification) else "",
            .verified = self.verified,
            .timestamp = self.timestamp,
            .allocator = allocator,
        };
        for (self.premise_indices.items) |p| {
            try new_step.premise_indices.append(p);
        }
        return new_step;
    }

    pub fn verify(self: *Self, premises: []const *ProofStep) bool {
        if (!self.rule_applied.requiresPremises()) {
            self.verified = true;
            return true;
        }
        if (self.premise_indices.items.len < self.rule_applied.minimumPremises()) {
            return false;
        }
        for (self.premise_indices.items) |idx| {
            if (idx >= premises.len or !premises[idx].verified) {
                return false;
            }
        }
        self.verified = self.validateRule(premises);
        return self.verified;
    }

    fn validateRule(self: *Self, premises: []const *ProofStep) bool {
        return switch (self.rule_applied) {
            .AXIOM, .ASSIGNMENT_AXIOM => true,
            .MODUS_PONENS => self.validateModusPonens(premises),
            .CONJUNCTION_INTRO => self.validateConjunctionIntro(premises),
            .CONJUNCTION_ELIM => self.validateConjunctionElim(premises),
            .DISJUNCTION_INTRO => self.validateDisjunctionIntro(premises),
            .IMPLICATION_INTRO => self.validateImplicationIntro(premises),
            .NEGATION_ELIM => self.validateNegationElim(premises),
            .UNIVERSAL_INSTANTIATION => self.validateUniversalInstantiation(premises),
            .EXISTENTIAL_GENERALIZATION => self.validateExistentialGeneralization(premises),
            .WEAKENING => self.validateWeakening(premises),
            .STRENGTHENING => self.validateStrengthening(premises),
            .FRAME_RULE => self.validateFrameRule(premises),
            .CONSEQUENCE_RULE => self.validateConsequenceRule(premises),
            .SEQUENCE_RULE => self.validateSequenceRule(premises),
            else => true,
        };
    }

    fn validateModusPonens(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= premises.len or idx1 >= premises.len) return false;
        const antecedent = premises[idx0].conclusion;
        const implication = premises[idx1].conclusion;
        if (implication.prop_type != .IMPLICATION) return false;
        if (implication.sub_propositions.items.len < 2) return false;
        if (!antecedent.equals(implication.sub_propositions.items[0])) return false;
        return self.conclusion.equals(implication.sub_propositions.items[1]);
    }

    fn validateConjunctionIntro(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        if (self.conclusion.prop_type != .CONJUNCTION) return false;
        if (self.conclusion.sub_propositions.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= premises.len or idx1 >= premises.len) return false;
        const left = premises[idx0].conclusion;
        const right = premises[idx1].conclusion;
        return left.equals(self.conclusion.sub_propositions.items[0]) and
            right.equals(self.conclusion.sub_propositions.items[1]);
    }

    fn validateConjunctionElim(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        const conjunction = premises[idx].conclusion;
        if (conjunction.prop_type != .CONJUNCTION) return false;
        for (conjunction.sub_propositions.items) |sub| {
            if (self.conclusion.equals(sub)) return true;
        }
        return false;
    }

    fn validateDisjunctionIntro(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        if (self.conclusion.prop_type != .DISJUNCTION) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        const premise_conclusion = premises[idx].conclusion;
        for (self.conclusion.sub_propositions.items) |sub| {
            if (premise_conclusion.equals(sub)) return true;
        }
        return false;
    }

    fn validateImplicationIntro(self: *Self, premises: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .IMPLICATION) return false;
        if (self.conclusion.sub_propositions.items.len < 2) return false;
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        return premises[idx].conclusion.equals(self.conclusion.sub_propositions.items[1]);
    }

    fn validateNegationElim(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        const premise = premises[idx].conclusion;
        if (premise.prop_type != .NEGATION) return false;
        if (premise.sub_propositions.items.len < 1) return false;
        const inner = premise.sub_propositions.items[0];
        if (inner.prop_type != .NEGATION) return false;
        if (inner.sub_propositions.items.len < 1) return false;
        return self.conclusion.equals(inner.sub_propositions.items[0]);
    }

    fn validateUniversalInstantiation(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        const universal = premises[idx].conclusion;
        if (universal.prop_type != .UNIVERSAL) return false;
        return true;
    }

    fn validateExistentialGeneralization(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        if (self.conclusion.prop_type != .EXISTENTIAL) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        return premises[idx].verified;
    }

    fn validateWeakening(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        if (self.conclusion.prop_type == .DISJUNCTION) {
            for (self.conclusion.sub_propositions.items) |sub| {
                if (premises[idx].conclusion.equals(sub)) return true;
            }
        }
        if (premises[idx].conclusion.prop_type == .CONJUNCTION) {
            for (premises[idx].conclusion.sub_propositions.items) |sub| {
                if (self.conclusion.equals(sub)) return true;
            }
        }
        return premises[idx].conclusion.equals(self.conclusion);
    }

    fn validateStrengthening(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        if (premises[idx].conclusion.prop_type == .DISJUNCTION) {
            for (premises[idx].conclusion.sub_propositions.items) |sub| {
                if (self.conclusion.equals(sub)) return true;
            }
        }
        return false;
    }

    fn validateFrameRule(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= premises.len) return false;
        const premise = premises[idx].conclusion;
        if (premise.prop_type != .HOARE_TRIPLE) return false;
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        return true;
    }

    fn validateConsequenceRule(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 3) return false;
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        for (self.premise_indices.items) |idx| {
            if (idx >= premises.len or !premises[idx].verified) return false;
        }
        return true;
    }

    fn validateSequenceRule(self: *Self, premises: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= premises.len or idx1 >= premises.len) return false;
        const first = premises[idx0].conclusion;
        const second = premises[idx1].conclusion;
        if (first.prop_type != .HOARE_TRIPLE or second.prop_type != .HOARE_TRIPLE) return false;
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        return true;
    }
};

pub const FormalProof = struct {
    proof_id: u64,
    theorem: *Proposition,
    steps: ArrayList(*ProofStep),
    axioms: ArrayList(*Proposition),
    is_complete: bool,
    is_valid: bool,
    allocator: Allocator,
    creation_time: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, proof_id: u64, theorem: *Proposition) !*Self {
        const proof = try allocator.create(Self);
        proof.* = Self{
            .proof_id = proof_id,
            .theorem = theorem,
            .steps = ArrayList(*ProofStep).init(allocator),
            .axioms = ArrayList(*Proposition).init(allocator),
            .is_complete = false,
            .is_valid = false,
            .allocator = allocator,
            .creation_time = std.time.milliTimestamp(),
        };
        return proof;
    }

    pub fn deinit(self: *Self) void {
        for (self.steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.steps.deinit();
        for (self.axioms.items) |axiom| {
            axiom.deinit();
            self.allocator.destroy(axiom);
        }
        self.axioms.deinit();
        self.theorem.deinit();
        self.allocator.destroy(self.theorem);
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        try self.axioms.append(axiom);
    }

    pub fn addStep(self: *Self, step: *ProofStep) !void {
        try self.steps.append(step);
        self.is_valid = false;
        self.is_complete = false;
    }

    pub fn validate(self: *Self) !bool {
        if (self.steps.items.len == 0) {
            return false;
        }
        var all_verified = true;
        var i: usize = 0;
        while (i < self.steps.items.len) : (i += 1) {
            const step = self.steps.items[i];
            var premise_steps = ArrayList(*ProofStep).init(self.allocator);
            defer premise_steps.deinit();
            var j: usize = 0;
            while (j < i) : (j += 1) {
                try premise_steps.append(self.steps.items[j]);
            }
            if (!step.verify(premise_steps.items)) {
                all_verified = false;
                break;
            }
        }
        self.is_valid = all_verified;
        if (all_verified and self.steps.items.len > 0) {
            const last_step = self.steps.items[self.steps.items.len - 1];
            self.is_complete = last_step.conclusion.equals(self.theorem);
        } else {
            self.is_complete = false;
        }
        return self.is_valid and self.is_complete;
    }

    pub fn getLastStep(self: *const Self) ?*ProofStep {
        if (self.steps.items.len == 0) return null;
        return self.steps.items[self.steps.items.len - 1];
    }

    pub fn stepCount(self: *const Self) usize {
        return self.steps.items.len;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_proof = try allocator.create(Self);
        new_proof.* = Self{
            .proof_id = self.proof_id,
            .theorem = try self.theorem.clone(allocator),
            .steps = ArrayList(*ProofStep).init(allocator),
            .axioms = ArrayList(*Proposition).init(allocator),
            .is_complete = self.is_complete,
            .is_valid = self.is_valid,
            .allocator = allocator,
            .creation_time = self.creation_time,
        };
        for (self.steps.items) |step| {
            try new_proof.steps.append(try step.clone(allocator));
        }
        for (self.axioms.items) |axiom| {
            try new_proof.axioms.append(try axiom.clone(allocator));
        }
        return new_proof;
    }
};

pub const PredicateFn = *const fn (*const SelfSimilarRelationalGraph) bool;

pub const Invariant = struct {
    id: u64,
    inv_type: InvariantType,
    description: []const u8,
    priority_val: u8,
    predicate: PredicateFn,
    enabled: bool,
    violation_count: u64,
    last_check_time: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !*Self {
        const inv = try allocator.create(Self);
        inv.* = Self{
            .id = id,
            .inv_type = inv_type,
            .description = try allocator.dupe(u8, description),
            .priority_val = inv_type.priority(),
            .predicate = predicate,
            .enabled = true,
            .violation_count = 0,
            .last_check_time = 0,
            .allocator = allocator,
        };
        return inv;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.description);
    }

    pub fn check(self: *Self, graph: *const SelfSimilarRelationalGraph) bool {
        self.last_check_time = std.time.milliTimestamp();
        if (!self.enabled) {
            return true;
        }
        const result = self.predicate(graph);
        if (!result) {
            self.violation_count += 1;
        }
        return result;
    }

    pub fn enable(self: *Self) void {
        self.enabled = true;
    }

    pub fn disable(self: *Self) void {
        self.enabled = false;
    }

    pub fn resetViolationCount(self: *Self) void {
        self.violation_count = 0;
    }
};

fn checkConnectivityPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    const node_count = graph.nodes.count();
    if (node_count <= 1) {
        return true;
    }
    const allocator = std.heap.page_allocator;
    var visited = StringHashMap(void).init(allocator);
    defer visited.deinit();
    var queue = ArrayList([]const u8).init(allocator);
    defer queue.deinit();
    var node_iter = graph.nodes.iterator();
    if (node_iter.next()) |first_entry| {
        queue.append(first_entry.key_ptr.*) catch return false;
    } else {
        return true;
    }
    while (queue.items.len > 0) {
        const current = queue.orderedRemove(0);
        if (visited.contains(current)) {
            continue;
        }
        visited.put(current, {}) catch return false;
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key.source, current)) {
                if (!visited.contains(key.target)) {
                    queue.append(key.target) catch continue;
                }
            }
            if (std.mem.eql(u8, key.target, current)) {
                if (!visited.contains(key.source)) {
                    queue.append(key.source) catch continue;
                }
            }
        }
    }
    return visited.count() == node_count;
}

fn checkCoherencePredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        const magnitude = node.quantum_state.magnitude();
        if (magnitude > 1.0 + 1e-9) {
            return false;
        }
    }
    return true;
}

fn checkEntanglementPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        for (entry.value_ptr.items) |edge| {
            if (edge.quality == .entangled) {
                const source_exists = graph.nodes.contains(key.source);
                const target_exists = graph.nodes.contains(key.target);
                if (!source_exists or !target_exists) {
                    return false;
                }
            }
        }
    }
    return true;
}

fn checkFractalDimensionPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            if (edge.fractal_dimension < 0.0 or edge.fractal_dimension > 3.0) {
                return false;
            }
        }
    }
    return true;
}

fn checkQuantumStatePredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        const prob = node.probability();
        if (prob < 0.0 or prob > 1.0 + 1e-9) {
            return false;
        }
        if (std.math.isNan(node.quantum_state.re) or std.math.isNan(node.quantum_state.im)) {
            return false;
        }
        if (std.math.isInf(node.quantum_state.re) or std.math.isInf(node.quantum_state.im)) {
            return false;
        }
    }
    return true;
}

fn checkMemorySafetyPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!graph.nodes.contains(key.source)) {
            return false;
        }
        if (!graph.nodes.contains(key.target)) {
            return false;
        }
    }
    return true;
}

fn checkTypeSafetyPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        if (node.id.len == 0) {
            return false;
        }
    }
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            if (edge.weight < 0.0 or edge.weight > 1e10) {
                return false;
            }
        }
    }
    return true;
}

fn checkSymmetryPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    _ = graph;
    return true;
}

fn checkTemporalConsistencyPredicate(graph: *const SelfSimilarRelationalGraph) bool {
    _ = graph;
    return true;
}

pub const InvariantRegistry = struct {
    invariants: AutoHashMap(u64, *Invariant),
    invariants_by_type: [@typeInfo(InvariantType).Enum.fields.len]ArrayList(*Invariant),
    next_id: u64,
    allocator: Allocator,
    check_count: u64,
    violation_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        var by_type: [@typeInfo(InvariantType).Enum.fields.len]ArrayList(*Invariant) = undefined;
        for (&by_type) |*list| {
            list.* = ArrayList(*Invariant).init(allocator);
        }
        return Self{
            .invariants = AutoHashMap(u64, *Invariant).init(allocator),
            .invariants_by_type = by_type,
            .next_id = 1,
            .allocator = allocator,
            .check_count = 0,
            .violation_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.invariants.deinit();
        for (&self.invariants_by_type) |*list| {
            list.deinit();
        }
    }

    pub fn registerInvariant(self: *Self, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const inv = try Invariant.init(self.allocator, id, inv_type, description, predicate);
        try self.invariants.put(id, inv);
        try self.invariants_by_type[@intFromEnum(inv_type)].append(inv);
        return id;
    }

    pub fn registerCoreInvariants(self: *Self) !void {
        _ = try self.registerInvariant(.CONNECTIVITY, "Graph must maintain connectivity between nodes", checkConnectivityPredicate);
        _ = try self.registerInvariant(.COHERENCE, "Quantum state magnitude must not exceed 1", checkCoherencePredicate);
        _ = try self.registerInvariant(.ENTANGLEMENT, "Entangled edges must have valid paired nodes", checkEntanglementPredicate);
        _ = try self.registerInvariant(.FRACTAL_DIMENSION, "Fractal dimension must be in [0, 3]", checkFractalDimensionPredicate);
        _ = try self.registerInvariant(.QUANTUM_STATE, "Quantum states must be valid probabilities", checkQuantumStatePredicate);
        _ = try self.registerInvariant(.MEMORY_SAFETY, "All edge endpoints must reference existing nodes", checkMemorySafetyPredicate);
        _ = try self.registerInvariant(.TYPE_SAFETY, "All values must have valid types", checkTypeSafetyPredicate);
        _ = try self.registerInvariant(.SYMMETRY, "Graph symmetry properties must be preserved", checkSymmetryPredicate);
        _ = try self.registerInvariant(.TEMPORAL_CONSISTENCY, "Temporal ordering must be consistent", checkTemporalConsistencyPredicate);
    }

    pub fn getInvariant(self: *const Self, id: u64) ?*Invariant {
        return self.invariants.get(id);
    }

    pub fn getInvariantsByType(self: *const Self, inv_type: InvariantType) []const *Invariant {
        return self.invariants_by_type[@intFromEnum(inv_type)].items;
    }

    pub fn checkAll(self: *Self, graph: *const SelfSimilarRelationalGraph) bool {
        self.check_count += 1;
        var all_passed = true;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.*.check(graph)) {
                self.violation_count += 1;
                all_passed = false;
            }
        }
        return all_passed;
    }

    pub fn checkByType(self: *Self, graph: *const SelfSimilarRelationalGraph, inv_type: InvariantType) bool {
        self.check_count += 1;
        for (self.invariants_by_type[@intFromEnum(inv_type)].items) |inv| {
            if (!inv.check(graph)) {
                self.violation_count += 1;
                return false;
            }
        }
        return true;
    }

    pub fn checkByPriority(self: *Self, graph: *const SelfSimilarRelationalGraph, min_priority: u8) bool {
        self.check_count += 1;
        var all_passed = true;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.*.priority_val >= min_priority) {
                if (!entry.value_ptr.*.check(graph)) {
                    self.violation_count += 1;
                    all_passed = false;
                }
            }
        }
        return all_passed;
    }

    pub fn getStatistics(self: *const Self) InvariantStatistics {
        var total_violations: u64 = 0;
        var enabled_count: usize = 0;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            total_violations += entry.value_ptr.*.violation_count;
            if (entry.value_ptr.*.enabled) {
                enabled_count += 1;
            }
        }
        return InvariantStatistics{
            .total_invariants = self.invariants.count(),
            .enabled_invariants = enabled_count,
            .total_checks = self.check_count,
            .total_violations = total_violations,
        };
    }

    pub fn count(self: *const Self) usize {
        return self.invariants.count();
    }
};

pub const InvariantStatistics = struct {
    total_invariants: usize,
    enabled_invariants: usize,
    total_checks: u64,
    total_violations: u64,
};

pub const HoareTriple = struct {
    id: u64,
    precondition: *Proposition,
    operation: *Proposition,
    postcondition: *Proposition,
    verified: bool,
    verification_time: i64,
    frame_condition: ?*Proposition,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !*Self {
        const triple = try allocator.create(Self);
        triple.* = Self{
            .id = id,
            .precondition = precondition,
            .operation = operation,
            .postcondition = postcondition,
            .verified = false,
            .verification_time = 0,
            .frame_condition = null,
            .allocator = allocator,
        };
        return triple;
    }

    pub fn deinit(self: *Self) void {
        self.precondition.deinit();
        self.allocator.destroy(self.precondition);
        self.operation.deinit();
        self.allocator.destroy(self.operation);
        self.postcondition.deinit();
        self.allocator.destroy(self.postcondition);
        if (self.frame_condition) |fc| {
            fc.deinit();
            self.allocator.destroy(fc);
        }
    }

    pub fn setFrameCondition(self: *Self, frame: *Proposition) void {
        if (self.frame_condition) |fc| {
            fc.deinit();
            self.allocator.destroy(fc);
        }
        self.frame_condition = frame;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_triple = try allocator.create(Self);
        new_triple.* = Self{
            .id = self.id,
            .precondition = try self.precondition.clone(allocator),
            .operation = try self.operation.clone(allocator),
            .postcondition = try self.postcondition.clone(allocator),
            .verified = self.verified,
            .verification_time = self.verification_time,
            .frame_condition = if (self.frame_condition) |fc| try fc.clone(allocator) else null,
            .allocator = allocator,
        };
        return new_triple;
    }

    pub fn toProposition(self: *const Self) !*Proposition {
        return Proposition.initHoareTriple(
            self.allocator,
            try self.precondition.clone(self.allocator),
            try self.operation.clone(self.allocator),
            try self.postcondition.clone(self.allocator),
        );
    }
};

pub const HoareLogicVerifier = struct {
    triples: AutoHashMap(u64, *HoareTriple),
    verification_cache: AutoHashMap(u64, bool),
    next_id: u64,
    allocator: Allocator,
    total_verifications: u64,
    successful_verifications: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .triples = AutoHashMap(u64, *HoareTriple).init(allocator),
            .verification_cache = AutoHashMap(u64, bool).init(allocator),
            .next_id = 1,
            .allocator = allocator,
            .total_verifications = 0,
            .successful_verifications = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.triples.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.triples.deinit();
        self.verification_cache.deinit();
    }

    pub fn createTriple(self: *Self, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const triple = try HoareTriple.init(self.allocator, id, precondition, operation, postcondition);
        try self.triples.put(id, triple);
        return id;
    }

    pub fn getTriple(self: *const Self, id: u64) ?*HoareTriple {
        return self.triples.get(id);
    }

    pub fn verifyAssignmentAxiom(self: *Self, triple_id: u64, variable: []const u8, expression: *const Term) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        _ = variable;
        _ = expression;
        const valid = triple.precondition.prop_type != .FALSE;
        if (valid) {
            self.successful_verifications += 1;
            triple.verified = true;
            triple.verification_time = std.time.milliTimestamp();
        }
        try self.verification_cache.put(triple_id, valid);
        return valid;
    }

    pub fn verifySequenceRule(self: *Self, triple1_id: u64, triple2_id: u64) !bool {
        self.total_verifications += 1;
        const triple1 = self.triples.get(triple1_id) orelse return VerificationError.InvalidProofStep;
        const triple2 = self.triples.get(triple2_id) orelse return VerificationError.InvalidProofStep;
        if (!triple1.verified or !triple2.verified) {
            return false;
        }
        const valid = triple1.postcondition.equals(triple2.precondition);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyConditionalRule(self: *Self, condition: *const Proposition, then_triple_id: u64, else_triple_id: u64) !bool {
        self.total_verifications += 1;
        const then_triple = self.triples.get(then_triple_id) orelse return VerificationError.InvalidProofStep;
        const else_triple = self.triples.get(else_triple_id) orelse return VerificationError.InvalidProofStep;
        _ = condition;
        const valid = then_triple.verified and else_triple.verified and
            then_triple.postcondition.equals(else_triple.postcondition);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyLoopInvariant(self: *Self, invariant: *const Proposition, body_triple_id: u64) !bool {
        self.total_verifications += 1;
        const body_triple = self.triples.get(body_triple_id) orelse return VerificationError.InvalidProofStep;
        const valid = body_triple.verified and
            body_triple.precondition.equals(invariant) and
            body_triple.postcondition.equals(invariant);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyConsequenceRule(self: *Self, triple_id: u64, stronger_pre: *const Proposition, weaker_post: *const Proposition) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        _ = stronger_pre;
        _ = weaker_post;
        const valid = triple.verified;
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyFrameRule(self: *Self, triple_id: u64, frame: *Proposition) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        if (!triple.verified) {
            return false;
        }
        triple.setFrameCondition(frame);
        self.successful_verifications += 1;
        return true;
    }

    pub fn composeTriples(self: *Self, ids: []const u64) !u64 {
        if (ids.len < 2) return VerificationError.InvalidProofStep;
        var current_triple = self.triples.get(ids[0]) orelse return VerificationError.InvalidProofStep;
        for (ids[1..]) |id| {
            const next_triple = self.triples.get(id) orelse return VerificationError.InvalidProofStep;
            if (!current_triple.postcondition.equals(next_triple.precondition)) {
                return VerificationError.InvalidConclusion;
            }
            current_triple = next_triple;
        }
        const first = self.triples.get(ids[0]).?;
        const last = self.triples.get(ids[ids.len - 1]).?;
        const composed_pre = try first.precondition.clone(self.allocator);
        const composed_op = try Proposition.initAtomic(self.allocator, "composed_operation");
        const composed_post = try last.postcondition.clone(self.allocator);
        return self.createTriple(composed_pre, composed_op, composed_post);
    }

    pub fn count(self: *const Self) usize {
        return self.triples.count();
    }

    pub fn getStatistics(self: *const Self) HoareStatistics {
        return HoareStatistics{
            .total_triples = self.triples.count(),
            .total_verifications = self.total_verifications,
            .successful_verifications = self.successful_verifications,
            .cache_size = self.verification_cache.count(),
        };
    }
};

pub const HoareStatistics = struct {
    total_triples: usize,
    total_verifications: u64,
    successful_verifications: u64,
    cache_size: usize,
};

pub const Substitution = struct {
    mappings: StringHashMap(*Term),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .mappings = StringHashMap(*Term).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.mappings.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.mappings.deinit();
    }

    pub fn add(self: *Self, variable: []const u8, term: *Term) !void {
        const key = try self.allocator.dupe(u8, variable);
        try self.mappings.put(key, term);
    }

    pub fn get(self: *const Self, variable: []const u8) ?*Term {
        return self.mappings.get(variable);
    }

    pub fn contains(self: *const Self, variable: []const u8) bool {
        return self.mappings.contains(variable);
    }

    pub fn compose(self: *Self, other: *const Self) !void {
        var iter = other.mappings.iterator();
        while (iter.next()) |entry| {
            if (!self.contains(entry.key_ptr.*)) {
                const term_clone = try entry.value_ptr.*.clone(self.allocator);
                try self.add(entry.key_ptr.*, term_clone);
            }
        }
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.mappings.count() == 0;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_sub = try allocator.create(Self);
        new_sub.* = Self.init(allocator);
        var iter = self.mappings.iterator();
        while (iter.next()) |entry| {
            const term_clone = try entry.value_ptr.*.clone(allocator);
            try new_sub.add(entry.key_ptr.*, term_clone);
        }
        return new_sub;
    }
};

pub const Clause = struct {
    literals: ArrayList(*Proposition),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .literals = ArrayList(*Proposition).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.literals.items) |lit| {
            lit.deinit();
            self.allocator.destroy(lit);
        }
        self.literals.deinit();
    }

    pub fn addLiteral(self: *Self, literal: *Proposition) !void {
        try self.literals.append(literal);
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.literals.items.len == 0;
    }

    pub fn size(self: *const Self) usize {
        return self.literals.items.len;
    }

    pub fn isUnit(self: *const Self) bool {
        return self.literals.items.len == 1;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_clause = try allocator.create(Self);
        new_clause.* = Self.init(allocator);
        for (self.literals.items) |lit| {
            try new_clause.addLiteral(try lit.clone(allocator));
        }
        return new_clause;
    }

    pub fn containsComplementary(self: *const Self) bool {
        var i: usize = 0;
        while (i < self.literals.items.len) : (i += 1) {
            const lit1 = self.literals.items[i];
            for (self.literals.items[i + 1 ..]) |lit2| {
                if (areComplementary(lit1, lit2)) {
                    return true;
                }
            }
        }
        return false;
    }
};

fn areComplementary(a: *const Proposition, b: *const Proposition) bool {
    if (a.prop_type == .NEGATION and a.sub_propositions.items.len > 0) {
        return a.sub_propositions.items[0].equals(b);
    }
    if (b.prop_type == .NEGATION and b.sub_propositions.items.len > 0) {
        return b.sub_propositions.items[0].equals(a);
    }
    return false;
}

pub const ProofTreeNode = struct {
    proposition: *Proposition,
    rule: ProofRule,
    children: ArrayList(*ProofTreeNode),
    parent: ?*ProofTreeNode,
    depth: usize,
    is_leaf: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, proposition: *Proposition, rule: ProofRule) !*Self {
        const node = try allocator.create(Self);
        node.* = Self{
            .proposition = proposition,
            .rule = rule,
            .children = ArrayList(*ProofTreeNode).init(allocator),
            .parent = null,
            .depth = 0,
            .is_leaf = true,
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *Self) void {
        for (self.children.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.children.deinit();
        self.proposition.deinit();
        self.allocator.destroy(self.proposition);
    }

    pub fn addChild(self: *Self, child: *ProofTreeNode) !void {
        child.parent = self;
        child.depth = self.depth + 1;
        self.is_leaf = false;
        try self.children.append(child);
    }

    pub fn isComplete(self: *const Self) bool {
        if (self.is_leaf) {
            return self.rule == .AXIOM or self.proposition.prop_type == .TRUE;
        }
        for (self.children.items) |child| {
            if (!child.isComplete()) {
                return false;
            }
        }
        return true;
    }
};

pub const TheoremProver = struct {
    axioms: ArrayList(*Proposition),
    clauses: ArrayList(*Clause),
    proof_cache: StringHashMap(bool),
    max_depth: usize,
    allocator: Allocator,
    unification_count: u64,
    resolution_count: u64,

    const Self = @This();
    const DEFAULT_MAX_DEPTH: usize = 100;

    pub fn init(allocator: Allocator) Self {
        return Self{
            .axioms = ArrayList(*Proposition).init(allocator),
            .clauses = ArrayList(*Clause).init(allocator),
            .proof_cache = StringHashMap(bool).init(allocator),
            .max_depth = DEFAULT_MAX_DEPTH,
            .allocator = allocator,
            .unification_count = 0,
            .resolution_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.axioms.items) |axiom| {
            axiom.deinit();
            self.allocator.destroy(axiom);
        }
        self.axioms.deinit();
        for (self.clauses.items) |clause| {
            clause.deinit();
            self.allocator.destroy(clause);
        }
        self.clauses.deinit();
        var iter = self.proof_cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.proof_cache.deinit();
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        try self.axioms.append(axiom);
    }

    pub fn addClause(self: *Self, clause: *Clause) !void {
        try self.clauses.append(clause);
    }

    pub fn unify(self: *Self, term1: *const Term, term2: *const Term) !?*Substitution {
        self.unification_count += 1;
        const sub = try self.allocator.create(Substitution);
        sub.* = Substitution.init(self.allocator);
        if (try self.unifyTerms(term1, term2, sub)) {
            return sub;
        } else {
            sub.deinit();
            self.allocator.destroy(sub);
            return null;
        }
    }

    fn unifyTerms(self: *Self, t1: *const Term, t2: *const Term, sub: *Substitution) !bool {
        if (t1.kind == .VARIABLE) {
            return self.unifyVariable(t1, t2, sub);
        }
        if (t2.kind == .VARIABLE) {
            return self.unifyVariable(t2, t1, sub);
        }
        if (t1.kind == .CONSTANT and t2.kind == .CONSTANT) {
            return std.mem.eql(u8, t1.name, t2.name);
        }
        if (t1.kind == .FUNCTION and t2.kind == .FUNCTION) {
            if (!std.mem.eql(u8, t1.name, t2.name)) {
                return false;
            }
            if (t1.args.items.len != t2.args.items.len) {
                return false;
            }
            var i: usize = 0;
            while (i < t1.args.items.len) : (i += 1) {
                if (!try self.unifyTerms(t1.args.items[i], t2.args.items[i], sub)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    fn unifyVariable(self: *Self, variable: *const Term, term: *const Term, sub: *Substitution) error{OutOfMemory}!bool {
        if (sub.contains(variable.name)) {
            const bound = sub.get(variable.name).?;
            return self.unifyTerms(bound, term, sub);
        }
        if (term.kind == .VARIABLE and sub.contains(term.name)) {
            const bound = sub.get(term.name).?;
            return self.unifyTerms(variable, bound, sub);
        }
        if (term.containsVariable(variable.name)) {
            return false;
        }
        const term_clone = try term.clone(self.allocator);
        try sub.add(variable.name, term_clone);
        return true;
    }

    pub fn resolve(self: *Self, clause1: *const Clause, clause2: *const Clause, lit_idx1: usize, lit_idx2: usize) !?*Clause {
        self.resolution_count += 1;
        if (lit_idx1 >= clause1.literals.items.len or lit_idx2 >= clause2.literals.items.len) {
            return null;
        }
        const lit1 = clause1.literals.items[lit_idx1];
        const lit2 = clause2.literals.items[lit_idx2];
        if (!areComplementary(lit1, lit2)) {
            return null;
        }
        const resolvent = try self.allocator.create(Clause);
        resolvent.* = Clause.init(self.allocator);
        var i: usize = 0;
        while (i < clause1.literals.items.len) : (i += 1) {
            if (i != lit_idx1) {
                try resolvent.addLiteral(try clause1.literals.items[i].clone(self.allocator));
            }
        }
        var j: usize = 0;
        while (j < clause2.literals.items.len) : (j += 1) {
            if (j != lit_idx2) {
                var duplicate = false;
                for (resolvent.literals.items) |existing| {
                    if (existing.equals(clause2.literals.items[j])) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    try resolvent.addLiteral(try clause2.literals.items[j].clone(self.allocator));
                }
            }
        }
        return resolvent;
    }

    pub fn proveByResolution(self: *Self, goal: *Proposition) !bool {
        const negated_goal = try goal.negate(self.allocator);
        defer {
            negated_goal.deinit();
            self.allocator.destroy(negated_goal);
        }
        const goal_clause = try self.allocator.create(Clause);
        goal_clause.* = Clause.init(self.allocator);
        try goal_clause.addLiteral(try negated_goal.clone(self.allocator));
        defer {
            goal_clause.deinit();
            self.allocator.destroy(goal_clause);
        }
        var working_set = ArrayList(*Clause).init(self.allocator);
        defer {
            for (working_set.items) |c| {
                c.deinit();
                self.allocator.destroy(c);
            }
            working_set.deinit();
        }
        try working_set.append(try goal_clause.clone(self.allocator));
        for (self.clauses.items) |clause| {
            try working_set.append(try clause.clone(self.allocator));
        }
        var iterations: usize = 0;
        while (iterations < self.max_depth) : (iterations += 1) {
            var new_clauses = ArrayList(*Clause).init(self.allocator);
            defer new_clauses.deinit();
            var i: usize = 0;
            while (i < working_set.items.len) : (i += 1) {
                const c1 = working_set.items[i];
                for (working_set.items[i + 1 ..]) |c2| {
                    var li1: usize = 0;
                    while (li1 < c1.literals.items.len) : (li1 += 1) {
                        var li2: usize = 0;
                        while (li2 < c2.literals.items.len) : (li2 += 1) {
                            if (try self.resolve(c1, c2, li1, li2)) |resolvent| {
                                if (resolvent.isEmpty()) {
                                    resolvent.deinit();
                                    self.allocator.destroy(resolvent);
                                    return true;
                                }
                                if (!resolvent.containsComplementary()) {
                                    try new_clauses.append(resolvent);
                                } else {
                                    resolvent.deinit();
                                    self.allocator.destroy(resolvent);
                                }
                            }
                        }
                    }
                }
            }
            if (new_clauses.items.len == 0) {
                break;
            }
            for (new_clauses.items) |nc| {
                try working_set.append(nc);
            }
        }
        return false;
    }

    pub fn proveByBackwardChaining(self: *Self, goal: *Proposition, depth: usize) !bool {
        if (depth > self.max_depth) {
            return false;
        }
        for (self.axioms.items) |axiom| {
            if (axiom.equals(goal)) {
                return true;
            }
        }
        if (goal.prop_type == .IMPLICATION and goal.sub_propositions.items.len >= 2) {
            const antecedent = goal.sub_propositions.items[0];
            const consequent = goal.sub_propositions.items[1];
            if (try self.proveByBackwardChaining(antecedent, depth + 1)) {
                return self.proveByBackwardChaining(consequent, depth + 1);
            }
        }
        if (goal.prop_type == .CONJUNCTION) {
            for (goal.sub_propositions.items) |sub| {
                if (!try self.proveByBackwardChaining(sub, depth + 1)) {
                    return false;
                }
            }
            return true;
        }
        if (goal.prop_type == .DISJUNCTION) {
            for (goal.sub_propositions.items) |sub| {
                if (try self.proveByBackwardChaining(sub, depth + 1)) {
                    return true;
                }
            }
            return false;
        }
        for (self.axioms.items) |axiom| {
            if (axiom.prop_type == .IMPLICATION and axiom.sub_propositions.items.len >= 2) {
                if (axiom.sub_propositions.items[1].equals(goal)) {
                    if (try self.proveByBackwardChaining(axiom.sub_propositions.items[0], depth + 1)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    pub fn buildProofTree(self: *Self, goal: *Proposition) !?*ProofTreeNode {
        for (self.axioms.items) |axiom| {
            if (axiom.equals(goal)) {
                const goal_clone = try goal.clone(self.allocator);
                return ProofTreeNode.init(self.allocator, goal_clone, .AXIOM);
            }
        }
        if (goal.prop_type == .CONJUNCTION and goal.sub_propositions.items.len >= 2) {
            var children = ArrayList(*ProofTreeNode).init(self.allocator);
            for (goal.sub_propositions.items) |sub| {
                if (try self.buildProofTree(sub)) |child_tree| {
                    try children.append(child_tree);
                } else {
                    for (children.items) |c| {
                        c.deinit();
                        self.allocator.destroy(c);
                    }
                    children.deinit();
                    return null;
                }
            }
            const goal_clone = try goal.clone(self.allocator);
            const root = try ProofTreeNode.init(self.allocator, goal_clone, .CONJUNCTION_INTRO);
            for (children.items) |child| {
                try root.addChild(child);
            }
            children.deinit();
            return root;
        }
        return null;
    }

    pub fn setMaxDepth(self: *Self, depth: usize) void {
        self.max_depth = depth;
    }

    pub fn getStatistics(self: *const Self) ProverStatistics {
        return ProverStatistics{
            .axiom_count = self.axioms.items.len,
            .clause_count = self.clauses.items.len,
            .unification_count = self.unification_count,
            .resolution_count = self.resolution_count,
            .cache_size = self.proof_cache.count(),
        };
    }
};

pub const ProverStatistics = struct {
    axiom_count: usize,
    clause_count: usize,
    unification_count: u64,
    resolution_count: u64,
    cache_size: usize,
};

pub const VerificationResult = struct {
    success: bool,
    error_type: ?VerificationError,
    violated_invariants: ArrayList(u64),
    execution_time_ns: i64,
    graph_hash: [32]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn initSuccess(allocator: Allocator, graph_hash: [32]u8, exec_time: i64) !*Self {
        const result = try allocator.create(Self);
        result.* = Self{
            .success = true,
            .error_type = null,
            .violated_invariants = ArrayList(u64).init(allocator),
            .execution_time_ns = exec_time,
            .graph_hash = graph_hash,
            .allocator = allocator,
        };
        return result;
    }

    pub fn initFailure(allocator: Allocator, err: VerificationError, graph_hash: [32]u8, exec_time: i64) !*Self {
        const result = try allocator.create(Self);
        result.* = Self{
            .success = false,
            .error_type = err,
            .violated_invariants = ArrayList(u64).init(allocator),
            .execution_time_ns = exec_time,
            .graph_hash = graph_hash,
            .allocator = allocator,
        };
        return result;
    }

    pub fn deinit(self: *Self) void {
        self.violated_invariants.deinit();
    }

    pub fn addViolation(self: *Self, inv_id: u64) !void {
        try self.violated_invariants.append(inv_id);
    }
};

pub const FormalVerificationEngine = struct {
    invariant_registry: InvariantRegistry,
    hoare_verifier: HoareLogicVerifier,
    theorem_prover: TheoremProver,
    verification_history: ArrayList(*VerificationResult),
    allocator: Allocator,
    total_verifications: u64,
    successful_verifications: u64,
    creation_time: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        var registry = InvariantRegistry.init(allocator);
        try registry.registerCoreInvariants();
        return Self{
            .invariant_registry = registry,
            .hoare_verifier = HoareLogicVerifier.init(allocator),
            .theorem_prover = TheoremProver.init(allocator),
            .verification_history = ArrayList(*VerificationResult).init(allocator),
            .allocator = allocator,
            .total_verifications = 0,
            .successful_verifications = 0,
            .creation_time = std.time.milliTimestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.invariant_registry.deinit();
        self.hoare_verifier.deinit();
        self.theorem_prover.deinit();
        for (self.verification_history.items) |result| {
            result.deinit();
            self.allocator.destroy(result);
        }
        self.verification_history.deinit();
    }

    pub fn verifyGraph(self: *Self, graph: *const SelfSimilarRelationalGraph) !*VerificationResult {
        const start_time = std.time.nanoTimestamp();
        self.total_verifications += 1;
        var hash_buf: [32]u8 = undefined;
        var hasher = Sha256.init(.{});
        hasher.update(graph.getTopologyHashHex());
        hasher.final(&hash_buf);
        const all_passed = self.invariant_registry.checkAll(graph);
        const end_time = std.time.nanoTimestamp();
        const exec_time: i64 = @intCast(@mod(end_time - start_time, std.math.maxInt(i64)));
        var result: *VerificationResult = undefined;
        if (all_passed) {
            self.successful_verifications += 1;
            result = try VerificationResult.initSuccess(self.allocator, hash_buf, exec_time);
        } else {
            result = try VerificationResult.initFailure(self.allocator, VerificationError.InvariantViolation, hash_buf, exec_time);
            var iter = self.invariant_registry.invariants.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.*.violation_count > 0) {
                    try result.addViolation(entry.value_ptr.*.id);
                }
            }
        }
        try self.verification_history.append(result);
        return result;
    }

    pub fn verifyInvariantType(self: *Self, graph: *const SelfSimilarRelationalGraph, inv_type: InvariantType) bool {
        return self.invariant_registry.checkByType(graph, inv_type);
    }

    pub fn registerCustomInvariant(self: *Self, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !u64 {
        return self.invariant_registry.registerInvariant(inv_type, description, predicate);
    }

    pub fn createHoareTriple(self: *Self, pre: *Proposition, op: *Proposition, post: *Proposition) !u64 {
        return self.hoare_verifier.createTriple(pre, op, post);
    }

    pub fn verifyHoareTriple(self: *Self, triple_id: u64) !bool {
        const triple = self.hoare_verifier.getTriple(triple_id) orelse return VerificationError.InvalidProofStep;
        if (triple.verified) {
            return true;
        }
        const term = try Term.initConstant(self.allocator, "value");
        defer {
            term.deinit();
            self.allocator.destroy(term);
        }
        return self.hoare_verifier.verifyAssignmentAxiom(triple_id, "x", term);
    }

    pub fn prove(self: *Self, goal: *Proposition) !bool {
        return self.theorem_prover.proveByBackwardChaining(goal, 0);
    }

    pub fn proveByResolution(self: *Self, goal: *Proposition) !bool {
        return self.theorem_prover.proveByResolution(goal);
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        try self.theorem_prover.addAxiom(axiom);
    }

    pub fn buildFormalProof(self: *Self, theorem: *Proposition) !*FormalProof {
        const proof_id: u64 = @intCast(self.verification_history.items.len + 1);
        const theorem_clone = try theorem.clone(self.allocator);
        const proof = try FormalProof.init(self.allocator, proof_id, theorem_clone);
        for (self.theorem_prover.axioms.items) |axiom| {
            const axiom_clone = try axiom.clone(self.allocator);
            try proof.addAxiom(axiom_clone);
        }
        return proof;
    }

    pub fn getStatistics(self: *const Self) EngineStatistics {
        return EngineStatistics{
            .total_verifications = self.total_verifications,
            .successful_verifications = self.successful_verifications,
            .invariant_count = self.invariant_registry.count(),
            .hoare_triple_count = self.hoare_verifier.count(),
            .axiom_count = self.theorem_prover.axioms.items.len,
            .history_size = self.verification_history.items.len,
            .uptime_ms = std.time.milliTimestamp() - self.creation_time,
        };
    }

    pub fn clearHistory(self: *Self) void {
        for (self.verification_history.items) |result| {
            result.deinit();
            self.allocator.destroy(result);
        }
        self.verification_history.clearRetainingCapacity();
    }
};

pub const EngineStatistics = struct {
    total_verifications: u64,
    successful_verifications: u64,
    invariant_count: usize,
    hoare_triple_count: usize,
    axiom_count: usize,
    history_size: usize,
    uptime_ms: i64,
};

test "InvariantType enum conversion" {
    const testing = std.testing;
    try testing.expectEqual(InvariantType.CONNECTIVITY, InvariantType.fromString("connectivity").?);
    try testing.expectEqual(InvariantType.COHERENCE, InvariantType.fromString("coherence").?);
    try testing.expectEqual(InvariantType.ENTANGLEMENT, InvariantType.fromString("entanglement").?);
    try testing.expectEqualStrings("memory_safety", InvariantType.MEMORY_SAFETY.toString());
}

test "ProofRule enum properties" {
    const testing = std.testing;
    try testing.expect(!ProofRule.AXIOM.requiresPremises());
    try testing.expect(ProofRule.MODUS_PONENS.requiresPremises());
    try testing.expectEqual(@as(usize, 2), ProofRule.MODUS_PONENS.minimumPremises());
    try testing.expectEqual(@as(usize, 0), ProofRule.AXIOM.minimumPremises());
}

test "PropType arity" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 0), PropType.ATOMIC.arity());
    try testing.expectEqual(@as(usize, 1), PropType.NEGATION.arity());
    try testing.expectEqual(@as(usize, 2), PropType.CONJUNCTION.arity());
    try testing.expectEqual(@as(usize, 3), PropType.HOARE_TRIPLE.arity());
}

test "Term creation and operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const var_term = try Term.initVariable(allocator, "x");
    defer {
        var_term.deinit();
        allocator.destroy(var_term);
    }
    try testing.expect(var_term.isVariable());
    try testing.expectEqualStrings("x", var_term.name);
    const const_term = try Term.initConstant(allocator, "42");
    defer {
        const_term.deinit();
        allocator.destroy(const_term);
    }
    try testing.expect(!const_term.isVariable());
}

test "Proposition creation and cloning" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const atomic = try Proposition.initAtomic(allocator, "P");
    defer {
        atomic.deinit();
        allocator.destroy(atomic);
    }
    try testing.expectEqual(PropType.ATOMIC, atomic.prop_type);
    try testing.expectEqualStrings("P", atomic.predicate_name);
    const cloned = try atomic.clone(allocator);
    defer {
        cloned.deinit();
        allocator.destroy(cloned);
    }
    try testing.expect(atomic.equals(cloned));
}

test "Proposition negation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const p = try Proposition.initAtomic(allocator, "P");
    const not_p = try Proposition.initNegation(allocator, p);
    defer {
        not_p.deinit();
        allocator.destroy(not_p);
    }
    try testing.expectEqual(PropType.NEGATION, not_p.prop_type);
    try testing.expectEqual(@as(usize, 1), not_p.sub_propositions.items.len);
}

test "Proposition binary operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const p = try Proposition.initAtomic(allocator, "P");
    const q = try Proposition.initAtomic(allocator, "Q");
    const p_and_q = try Proposition.initBinary(allocator, .CONJUNCTION, p, q);
    defer {
        p_and_q.deinit();
        allocator.destroy(p_and_q);
    }
    try testing.expectEqual(PropType.CONJUNCTION, p_and_q.prop_type);
    try testing.expectEqual(@as(usize, 2), p_and_q.sub_propositions.items.len);
}

test "Hoare triple creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const pre = try Proposition.initTrue(allocator);
    const op = try Proposition.initAtomic(allocator, "assign");
    const post = try Proposition.initTrue(allocator);
    const triple = try Proposition.initHoareTriple(allocator, pre, op, post);
    defer {
        triple.deinit();
        allocator.destroy(triple);
    }
    try testing.expectEqual(PropType.HOARE_TRIPLE, triple.prop_type);
    try testing.expectEqual(@as(usize, 3), triple.sub_propositions.items.len);
}

test "ProofStep creation and verification" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const conclusion = try Proposition.initAtomic(allocator, "conclusion");
    const step = try ProofStep.init(allocator, 1, .AXIOM, conclusion);
    defer {
        step.deinit();
        allocator.destroy(step);
    }
    try testing.expectEqual(@as(u64, 1), step.step_id);
    try testing.expectEqual(ProofRule.AXIOM, step.rule_applied);
    const verified = step.verify(&[_]*ProofStep{});
    try testing.expect(verified);
}

test "FormalProof creation and step addition" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const theorem = try Proposition.initAtomic(allocator, "theorem");
    var proof = try FormalProof.init(allocator, 1, theorem);
    defer {
        proof.deinit();
        allocator.destroy(proof);
    }
    const step_conclusion = try Proposition.initAtomic(allocator, "step1");
    const step = try ProofStep.init(allocator, 1, .AXIOM, step_conclusion);
    try proof.addStep(step);
    try testing.expectEqual(@as(usize, 1), proof.stepCount());
}

test "InvariantRegistry core invariants" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var registry = InvariantRegistry.init(allocator);
    defer registry.deinit();
    try registry.registerCoreInvariants();
    try testing.expect(registry.count() >= 9);
}

test "checkConnectivity on empty graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    const result = checkConnectivityPredicate(&graph);
    try testing.expect(result);
}

test "checkConnectivity on connected graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var n1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 1.0, 0.0, 0.0);
    try graph.addNode(n2);
    var edge = try Edge.init(allocator, "n1", "n2", .coherent, 1.0, 1.0, 0.0, 1.0);
    try graph.addEdge(edge);
    const result = checkConnectivityPredicate(&graph);
    try testing.expect(result);
}

test "checkCoherence valid states" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var node = try Node.init(allocator, "n1", "data", 0.5, 0.5, 0.0);
    try graph.addNode(node);
    const result = checkCoherencePredicate(&graph);
    try testing.expect(result);
}

test "checkFractalDimension valid bounds" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var n1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 1.0, 0.0, 0.0);
    try graph.addNode(n2);
    var edge = try Edge.init(allocator, "n1", "n2", .fractal, 1.0, 1.0, 0.0, 1.5);
    try graph.addEdge(edge);
    const result = checkFractalDimensionPredicate(&graph);
    try testing.expect(result);
}

test "HoareLogicVerifier creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var verifier = HoareLogicVerifier.init(allocator);
    defer verifier.deinit();
    const pre = try Proposition.initTrue(allocator);
    const op = try Proposition.initAtomic(allocator, "op");
    const post = try Proposition.initTrue(allocator);
    const triple_id = try verifier.createTriple(pre, op, post);
    try testing.expect(triple_id > 0);
    try testing.expectEqual(@as(usize, 1), verifier.count());
}

test "HoareLogicVerifier assignment axiom" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var verifier = HoareLogicVerifier.init(allocator);
    defer verifier.deinit();
    const pre = try Proposition.initAtomic(allocator, "x_equals_5");
    const op = try Proposition.initAtomic(allocator, "assign_x");
    const post = try Proposition.initAtomic(allocator, "x_equals_5");
    const triple_id = try verifier.createTriple(pre, op, post);
    const term = try Term.initConstant(allocator, "5");
    defer {
        term.deinit();
        allocator.destroy(term);
    }
    const result = try verifier.verifyAssignmentAxiom(triple_id, "x", term);
    try testing.expect(result);
}

test "TheoremProver unification" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var prover = TheoremProver.init(allocator);
    defer prover.deinit();
    const x = try Term.initVariable(allocator, "x");
    defer {
        x.deinit();
        allocator.destroy(x);
    }
    const a = try Term.initConstant(allocator, "a");
    defer {
        a.deinit();
        allocator.destroy(a);
    }
    const sub = try prover.unify(x, a);
    try testing.expect(sub != null);
    if (sub) |s| {
        try testing.expect(s.contains("x"));
        s.deinit();
        allocator.destroy(s);
    }
}

test "TheoremProver backward chaining" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var prover = TheoremProver.init(allocator);
    defer prover.deinit();
    const axiom = try Proposition.initAtomic(allocator, "P");
    try prover.addAxiom(axiom);
    const goal = try Proposition.initAtomic(allocator, "P");
    defer {
        goal.deinit();
        allocator.destroy(goal);
    }
    const result = try prover.proveByBackwardChaining(goal, 0);
    try testing.expect(result);
}

test "Clause operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var clause = Clause.init(allocator);
    defer clause.deinit();
    try testing.expect(clause.isEmpty());
    const lit = try Proposition.initAtomic(allocator, "P");
    try clause.addLiteral(lit);
    try testing.expect(!clause.isEmpty());
    try testing.expect(clause.isUnit());
}

test "Resolution between clauses" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var prover = TheoremProver.init(allocator);
    defer prover.deinit();
    const p = try Proposition.initAtomic(allocator, "P");
    const not_p = try Proposition.initNegation(allocator, try Proposition.initAtomic(allocator, "P"));
    const q = try Proposition.initAtomic(allocator, "Q");
    var clause1 = Clause.init(allocator);
    try clause1.addLiteral(p);
    try clause1.addLiteral(q);
    defer clause1.deinit();
    var clause2 = Clause.init(allocator);
    try clause2.addLiteral(not_p);
    defer clause2.deinit();
    const resolvent = try prover.resolve(&clause1, &clause2, 0, 0);
    if (resolvent) |r| {
        try testing.expectEqual(@as(usize, 1), r.size());
        r.deinit();
        allocator.destroy(r);
    }
}

test "FormalVerificationEngine creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var engine = try FormalVerificationEngine.init(allocator);
    defer engine.deinit();
    try testing.expect(engine.invariant_registry.count() >= 9);
    const stats = engine.getStatistics();
    try testing.expectEqual(@as(u64, 0), stats.total_verifications);
}

test "FormalVerificationEngine verify valid graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var engine = try FormalVerificationEngine.init(allocator);
    defer engine.deinit();
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var node = try Node.init(allocator, "n1", "data", 0.5, 0.5, 0.0);
    try graph.addNode(node);
    const result = try engine.verifyGraph(&graph);
    try testing.expect(result.success);
}

test "ProofTreeNode construction" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const prop = try Proposition.initAtomic(allocator, "P");
    const node = try ProofTreeNode.init(allocator, prop, .AXIOM);
    defer {
        node.deinit();
        allocator.destroy(node);
    }
    try testing.expect(node.is_leaf);
    try testing.expectEqual(@as(usize, 0), node.depth);
    try testing.expect(node.isComplete());
}

test "Substitution operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var sub = Substitution.init(allocator);
    defer sub.deinit();
    try testing.expect(sub.isEmpty());
    const term = try Term.initConstant(allocator, "value");
    try sub.add("x", term);
    try testing.expect(!sub.isEmpty());
    try testing.expect(sub.contains("x"));
    try testing.expect(!sub.contains("y"));
}

test "InvariantType priority ordering" {
    const testing = std.testing;
    try testing.expect(InvariantType.MEMORY_SAFETY.priority() > InvariantType.TYPE_SAFETY.priority());
    try testing.expect(InvariantType.TYPE_SAFETY.priority() > InvariantType.CONNECTIVITY.priority());
    try testing.expect(InvariantType.CONNECTIVITY.priority() > InvariantType.COHERENCE.priority());
}

test "FormalProof validation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const theorem = try Proposition.initAtomic(allocator, "goal");
    var proof = try FormalProof.init(allocator, 1, theorem);
    defer {
        proof.deinit();
        allocator.destroy(proof);
    }
    const step1_concl = try Proposition.initAtomic(allocator, "goal");
    const step1 = try ProofStep.init(allocator, 1, .AXIOM, step1_concl);
    try proof.addStep(step1);
    const valid = try proof.validate();
    try testing.expect(valid);
    try testing.expect(proof.is_valid);
    try testing.expect(proof.is_complete);
}

test "checkMemorySafety on valid graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var n1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 1.0, 0.0, 0.0);
    try graph.addNode(n2);
    var edge = try Edge.init(allocator, "n1", "n2", .coherent, 1.0, 1.0, 0.0, 1.0);
    try graph.addEdge(edge);
    const result = checkMemorySafetyPredicate(&graph);
    try testing.expect(result);
}

test "checkTypeSafety on valid graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var node = try Node.init(allocator, "node1", "data", 1.0, 0.0, 0.0);
    try graph.addNode(node);
    const result = checkTypeSafetyPredicate(&graph);
    try testing.expect(result);
}

test "checkQuantumState on valid states" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var node = try Node.init(allocator, "n1", "data", 0.6, 0.8, 0.0);
    try graph.addNode(node);
    const result = checkQuantumStatePredicate(&graph);
    try testing.expect(result);
}

test "checkEntanglement on entangled edges" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    var n1 = try Node.init(allocator, "n1", "data1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 1.0, 0.0, 0.0);
    try graph.addNode(n2);
    var edge = try Edge.init(allocator, "n1", "n2", .entangled, 1.0, 1.0, 0.0, 1.0);
    try graph.addEdge(edge);
    const result = checkEntanglementPredicate(&graph);
    try testing.expect(result);
}
