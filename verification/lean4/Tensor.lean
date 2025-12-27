import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Vector
import Mathlib.Tactic

namespace Tensor

def shapeSize : List Nat → Nat
  | [] => 1
  | d :: ds => d * shapeSize ds

theorem shapeSizePositive (shape : List Nat) (h : ∀ d ∈ shape, d > 0) :
  shapeSize shape > 0 := by
  induction shape with
  | nil => norm_num [shapeSize]
  | cons d ds ih =>
    simp [shapeSize]
    have hd := h d (List.mem_cons_self d ds)
    have hds : ∀ x ∈ ds, x > 0 := fun x hx => h x (List.mem_cons_of_mem d hx)
    have ih_res := ih hds
    omega

def computeStrides (shape : List Nat) : List Nat :=
  match shape with
  | [] => []
  | [x] => [1]
  | x :: xs =>
    let tailStrides := computeStrides xs
    match tailStrides with
    | [] => [1]
    | s :: _ => (s * x) :: tailStrides

structure TensorSpec (shape : List Nat) where
  dataVec : Vector Nat (shapeSize shape)
  refcount : Nat
  validRefcount : refcount > 0

def tensorInit (shape : List Nat) (hPos : ∀ d ∈ shape, d > 0) : TensorSpec shape :=
  ⟨Vector.replicate (shapeSize shape) 0, 1, Nat.zero_lt_one⟩

def tensorRetain {shape : List Nat} (t : TensorSpec shape) : TensorSpec shape :=
  ⟨t.dataVec, t.refcount + 1, Nat.zero_lt_succ t.refcount⟩

def tensorRelease {shape : List Nat} (t : TensorSpec shape) : Option (TensorSpec shape) :=
  match t.refcount with
  | 0 => none
  | 1 => none
  | n+2 => some ⟨t.dataVec, n+1, Nat.zero_lt_succ n⟩

def computeFlatIndex (indices strides : List Nat) : Nat :=
  match indices, strides with
  | [], [] => 0
  | i :: is, s :: ss => i * s + computeFlatIndex is ss
  | _, _ => 0

def tensorGet {shape : List Nat} (t : TensorSpec shape) (indices : List Nat)
  (hLen : indices.length = shape.length)
  (hBounds : ∀ i : Fin indices.length, indices.get i < shape.get ⟨i, by omega⟩) : Nat :=
  let strides := computeStrides shape
  let flatIdx := computeFlatIndex indices strides
  if h : flatIdx < shapeSize shape then
    t.dataVec.get ⟨flatIdx, h⟩
  else
    0

def tensorSet {shape : List Nat} (t : TensorSpec shape) (indices : List Nat) (value : Nat)
  (hLen : indices.length = shape.length)
  (hBounds : ∀ i : Fin indices.length, indices.get i < shape.get ⟨i, by omega⟩) : TensorSpec shape :=
  let strides := computeStrides shape
  let flatIdx := computeFlatIndex indices strides
  if h : flatIdx < shapeSize shape then
    ⟨t.dataVec.set ⟨flatIdx, h⟩ value, t.refcount, t.validRefcount⟩
  else
    t

def tensorFill {shape : List Nat} (t : TensorSpec shape) (value : Nat) : TensorSpec shape :=
  ⟨Vector.replicate (shapeSize shape) value, t.refcount, t.validRefcount⟩

def tensorAddPointwise {shape : List Nat} (t1 t2 : TensorSpec shape) : TensorSpec shape :=
  ⟨t1.dataVec.zipWith t2.dataVec (· + ·), t1.refcount, t1.validRefcount⟩

def tensorSubPointwise {shape : List Nat} (t1 t2 : TensorSpec shape) : TensorSpec shape :=
  ⟨t1.dataVec.zipWith t2.dataVec (· - ·), t1.refcount, t1.validRefcount⟩

def tensorMulPointwise {shape : List Nat} (t1 t2 : TensorSpec shape) : TensorSpec shape :=
  ⟨t1.dataVec.zipWith t2.dataVec (· * ·), t1.refcount, t1.validRefcount⟩

def tensorScalarAdd {shape : List Nat} (t : TensorSpec shape) (scalar : Nat) : TensorSpec shape :=
  ⟨t.dataVec.map (· + scalar), t.refcount, t.validRefcount⟩

def tensorScalarMul {shape : List Nat} (t : TensorSpec shape) (scalar : Nat) : TensorSpec shape :=
  ⟨t.dataVec.map (· * scalar), t.refcount, t.validRefcount⟩

def tensorSumAll {shape : List Nat} (t : TensorSpec shape) : Nat :=
  t.dataVec.toList.foldl (· + ·) 0

def tensorMaxElement {shape : List Nat} (t : TensorSpec shape) (hPos : shapeSize shape > 0) : Nat :=
  t.dataVec.toList.foldl max 0

def tensorMinElement {shape : List Nat} (t : TensorSpec shape) (hPos : shapeSize shape > 0) : Nat :=
  t.dataVec.toList.foldl min 1000000000

theorem tensorRetainIncreasesRefcount {shape : List Nat} (t : TensorSpec shape) :
  (tensorRetain t).refcount = t.refcount + 1 := by
  rfl

theorem tensorAddComm {shape : List Nat} (t1 t2 : TensorSpec shape) :
  (tensorAddPointwise t1 t2).dataVec = (tensorAddPointwise t2 t1).dataVec := by
  simp [tensorAddPointwise]
  ext i
  simp [Vector.zipWith]
  ring

theorem tensorAddAssoc {shape : List Nat} (t1 t2 t3 : TensorSpec shape) :
  (tensorAddPointwise (tensorAddPointwise t1 t2) t3).dataVec =
  (tensorAddPointwise t1 (tensorAddPointwise t2 t3)).dataVec := by
  simp [tensorAddPointwise]
  ext i
  simp [Vector.zipWith]
  ring

theorem tensorMulComm {shape : List Nat} (t1 t2 : TensorSpec shape) :
  (tensorMulPointwise t1 t2).dataVec = (tensorMulPointwise t2 t1).dataVec := by
  simp [tensorMulPointwise]
  ext i
  simp [Vector.zipWith]
  ring

theorem tensorMulAssoc {shape : List Nat} (t1 t2 t3 : TensorSpec shape) :
  (tensorMulPointwise (tensorMulPointwise t1 t2) t3).dataVec =
  (tensorMulPointwise t1 (tensorMulPointwise t2 t3)).dataVec := by
  simp [tensorMulPointwise]
  ext i
  simp [Vector.zipWith]
  ring

theorem tensorScalarMulDistributive {shape : List Nat} (t1 t2 : TensorSpec shape) (s : Nat) :
  (tensorScalarMul (tensorAddPointwise t1 t2) s).dataVec =
  (tensorAddPointwise (tensorScalarMul t1 s) (tensorScalarMul t2 s)).dataVec := by
  simp [tensorScalarMul, tensorAddPointwise]
  ext i
  simp [Vector.map, Vector.zipWith]
  ring

theorem tensorFillAllEqual {shape : List Nat} (t : TensorSpec shape) (v : Nat)
  (i j : Fin (shapeSize shape)) :
  (tensorFill t v).dataVec.get i = (tensorFill t v).dataVec.get j := by
  simp [tensorFill, Vector.replicate]

theorem tensorSumAdd {shape : List Nat} (t1 t2 : TensorSpec shape) :
  tensorSumAll (tensorAddPointwise t1 t2) =
  tensorSumAll t1 + tensorSumAll t2 := by
  unfold tensorSumAll tensorAddPointwise
  rw [Vector.toList_zipWith]
  induction t1.dataVec.toList, t2.dataVec.toList using List.rec₂ with
  | nil => rfl
  | cons a as b bs ih =>
    unfold List.foldl List.zipWith
    rw [List.foldl_cons, List.foldl_cons, List.foldl_cons]
    ring_nf
    have : List.foldl (· + ·) 0 (List.zipWith (· + ·) as bs) = 
           List.foldl (· + ·) 0 as + List.foldl (· + ·) 0 bs := ih
    omega

theorem tensorSumScalarMul {shape : List Nat} (t : TensorSpec shape) (s : Nat) :
  tensorSumAll (tensorScalarMul t s) = s * tensorSumAll t := by
  unfold tensorSumAll tensorScalarMul
  rw [Vector.toList_map]
  induction t.dataVec.toList with
  | nil => rfl
  | cons a as ih =>
    unfold List.map List.foldl
    rw [List.foldl_cons, List.foldl_cons]
    have h := ih
    ring_nf
    omega

def reshapeValid (oldShape newShape : List Nat) : Bool :=
  shapeSize oldShape = shapeSize newShape

theorem reshapePreservesSize (oldShape newShape : List Nat)
  (h : reshapeValid oldShape newShape = true) :
  shapeSize oldShape = shapeSize newShape := by
  simp [reshapeValid] at h
  exact h

def broadcastCompatible : List Nat → List Nat → Bool
  | [], [] => true
  | [], _ :: ds => broadcastCompatible [] ds
  | _ :: ds, [] => broadcastCompatible ds []
  | d1 :: ds1, d2 :: ds2 =>
    (d1 = d2 ∨ d1 = 1 ∨ d2 = 1) && broadcastCompatible ds1 ds2

def sliceInBounds (shape : List Nat) (start finish : List Nat) : Bool :=
  shape.length = start.length && start.length = finish.length &&
  (List.zip start (List.zip finish shape)).all fun (s, e, d) => s ≤ e && e ≤ d

def transposeAxesValid (shape axes : List Nat) : Bool :=
  shape.length = axes.length && axes.Nodup

def matmulShapesCompatible : List Nat → List Nat → Bool
  | [m, n], [n', k] => n = n'
  | _, _ => false

def computeMatmulOutputShape : (s1 s2 : List Nat) → matmulShapesCompatible s1 s2 = true → List Nat
  | [m, n], [n', k], _ => [m, k]
  | _, _, _ => []

def conv2dShapesValid (inputShape kernelShape : List Nat) (stride padding : Nat) : Bool :=
  match inputShape, kernelShape with
  | [batch, inH, inW, inC], [kH, kW, kInC, kOutC] =>
    inC = kInC && stride > 0 && kH ≤ inH + 2 * padding && kW ≤ inW + 2 * padding
  | _, _ => false

def pool2dShapesValid (inputShape : List Nat) (poolH poolW stride : Nat) : Bool :=
  match inputShape with
  | [batch, inH, inW, channels] =>
    stride > 0 && poolH ≤ inH && poolW ≤ inW
  | _ => false

theorem broadcastSymmetric (s1 s2 : List Nat)
  (h : broadcastCompatible s1 s2 = true) :
  broadcastCompatible s2 s1 = true := by
  induction s1, s2 using List.rec₂ with
  | nil => 
    unfold broadcastCompatible
    induction s2 with
    | nil => rfl
    | cons d ds ih => 
      unfold broadcastCompatible at h ⊢
      exact ih h
  | cons a as b bs ih =>
    unfold broadcastCompatible at h ⊢
    cases Decidable.em (a = b) with
    | inl hab =>
      rw [hab] at h ⊢
      have hrest : broadcastCompatible as bs = true := by
        unfold broadcastCompatible at h
        exact h.2
      constructor
      · left; rfl
      · exact ih hrest
    | inr hab =>
      cases Decidable.em (a = 1) with
      | inl ha1 =>
        rw [ha1] at h
        have hrest : broadcastCompatible as bs = true := h.2
        constructor
        · right; right; rfl  
        · exact ih hrest
      | inr ha1 =>
        cases Decidable.em (b = 1) with
        | inl hb1 =>
          rw [hb1]
          constructor
          · right; left; rfl
          · have hrest : broadcastCompatible as bs = true := h.2
            exact ih hrest
        | inr hb1 =>
          exfalso
          unfold broadcastCompatible at h
          have hcontra := h.1
          cases hcontra with
          | inl heq => exact hab heq
          | inr hor => 
            cases hor with
            | inl h1 => exact ha1 h1
            | inr h2 => exact hb1 h2

theorem matmulOutputSizePositive (s1 s2 : List Nat) (h : matmulShapesCompatible s1 s2 = true) :
  shapeSize (computeMatmulOutputShape s1 s2 h) > 0 := by
  cases s1 with
  | nil => simp [matmulShapesCompatible] at h
  | cons m s1' =>
    cases s1' with
    | nil => simp [matmulShapesCompatible] at h
    | cons n s1'' =>
      cases s2 with
      | nil => simp [matmulShapesCompatible] at h
      | cons n' s2' =>
        cases s2' with
        | nil => simp [matmulShapesCompatible] at h
        | cons k s2'' =>
          simp [computeMatmulOutputShape, shapeSize]
          omega

end Tensor
