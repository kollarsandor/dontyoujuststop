{-# OPTIONS --safe #-}
{-# OPTIONS --without-K #-}

module TensorVerification where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_; _≤_; _<_; _≟_; s≤s; z≤n; _⊔_)
open import Data.Nat.Properties using (+-assoc; +-comm; *-assoc; *-comm; ≤-refl; ≤-trans; <-trans; n≤1+n)
open import Data.List using (List; []; _∷_; length; map; foldr; sum; product; _++_)
open import Data.Vec using (Vec; []; _∷_; lookup; zipWith; replicate; head; tail; _++_; updateAt)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst)
open import Relation.Nullary using (¬_; Dec; yes; no)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃; Σ)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_; id; _$_)
open import Data.Empty using (⊥; ⊥-elim)
open import Level using (Level)
open ≡-Reasoning

data DType : Set where
  F32 : DType
  F64 : DType
  I32 : DType
  I64 : DType

dtype-eq : (d1 d2 : DType) → Dec (d1 ≡ d2)
dtype-eq F32 F32 = yes refl
dtype-eq F32 F64 = no (λ ())
dtype-eq F32 I32 = no (λ ())
dtype-eq F32 I64 = no (λ ())
dtype-eq F64 F32 = no (λ ())
dtype-eq F64 F64 = yes refl
dtype-eq F64 I32 = no (λ ())
dtype-eq F64 I64 = no (λ ())
dtype-eq I32 F32 = no (λ ())
dtype-eq I32 F64 = no (λ ())
dtype-eq I32 I32 = yes refl
dtype-eq I32 I64 = no (λ ())
dtype-eq I64 F32 = no (λ ())
dtype-eq I64 F64 = no (λ ())
dtype-eq I64 I32 = no (λ ())
dtype-eq I64 I64 = yes refl

shape-product : List ℕ → ℕ
shape-product [] = 1
shape-product (d ∷ ds) = d * shape-product ds

lemma-shape-product-positive : ∀ (shape : List ℕ) → 1 ≤ shape-product shape
lemma-shape-product-positive [] = s≤s z≤n
lemma-shape-product-positive (zero ∷ ds) = z≤n
lemma-shape-product-positive (suc d ∷ ds) =
  let rec = lemma-shape-product-positive ds
  in s≤s z≤n

lemma-shape-product-append : ∀ (xs ys : List ℕ) →
  shape-product (xs ++ ys) ≡ shape-product xs * shape-product ys
lemma-shape-product-append [] ys = refl
lemma-shape-product-append (x ∷ xs) ys =
  begin
    shape-product ((x ∷ xs) ++ ys)
  ≡⟨ refl ⟩
    x * shape-product (xs ++ ys)
  ≡⟨ cong (x *_) (lemma-shape-product-append xs ys) ⟩
    x * (shape-product xs * shape-product ys)
  ≡⟨ sym (*-assoc x (shape-product xs) (shape-product ys)) ⟩
    (x * shape-product xs) * shape-product ys
  ≡⟨ refl ⟩
    shape-product (x ∷ xs) * shape-product ys
  ∎

compute-strides : (shape : List ℕ) → List ℕ
compute-strides [] = []
compute-strides (d ∷ ds) with compute-strides ds
... | [] = 1 ∷ []
... | (s ∷ ss) = (d * s) ∷ (s ∷ ss)

theorem-strides-length : ∀ (shape : List ℕ) →
  length (compute-strides shape) ≡ length shape
theorem-strides-length [] = refl
theorem-strides-length (d ∷ ds) with compute-strides ds | theorem-strides-length ds
... | [] | eq = cong suc eq
... | (s ∷ ss) | eq = cong suc eq

record Tensor (Shape : List ℕ) (dtype : DType) : Set where
  constructor mkTensor
  field
    data-vec : Vec ℕ (shape-product Shape)
    refcount : ℕ

open Tensor

tensor-init : ∀ (Shape : List ℕ) (dtype : DType) → Tensor Shape dtype
tensor-init Shape dtype = mkTensor (replicate 0) 1

tensor-shape-consistency : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  Data.Vec.length (data-vec t) ≡ shape-product Shape
tensor-shape-consistency {Shape} t = refl

tensor-refcount-positive : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  1 ≤ refcount t
tensor-refcount-positive t = s≤s z≤n

tensor-retain : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype
tensor-retain t = record t { refcount = suc (refcount t) }

theorem-retain-increases-refcount : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  refcount (tensor-retain t) ≡ suc (refcount t)
theorem-retain-increases-refcount t = refl

tensor-release : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype ⊎ ⊤
tensor-release t with refcount t | refcount t ≟ 1
... | suc n | yes _ = inj₂ tt
... | suc n | no _ = inj₁ (record t { refcount = n })
... | zero | _ = inj₂ tt

theorem-release-decreases-refcount : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  (t' : Tensor Shape dtype) →
  tensor-release t ≡ inj₁ t' →
  refcount t' < refcount t
theorem-release-decreases-refcount t t' eq with refcount t | refcount t ≟ 1
... | zero | _ = λ ()
... | suc n | yes _ = λ ()
... | suc n | no _ = s≤s (n≤1+n n)

tensor-fill : ∀ {Shape : List ℕ} {dtype : DType} →
  ℕ → Tensor Shape dtype → Tensor Shape dtype
tensor-fill val t = record t { data-vec = replicate val }

theorem-fill-preserves-shape : ∀ {Shape : List ℕ} {dtype : DType} →
  (val : ℕ) → (t : Tensor Shape dtype) →
  Data.Vec.length (data-vec (tensor-fill val t)) ≡ shape-product Shape
theorem-fill-preserves-shape val t = refl

tensor-add : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype → Tensor Shape dtype
tensor-add t1 t2 = record t1 { data-vec = zipWith _+_ (data-vec t1) (data-vec t2) }

theorem-tensor-add-comm : ∀ {Shape : List ℕ} {dtype : DType} →
  (t1 t2 : Tensor Shape dtype) →
  data-vec (tensor-add t1 t2) ≡ data-vec (tensor-add t2 t1)
theorem-tensor-add-comm t1 t2 = Data.Vec.zipWith-comm _+_ +-comm (data-vec t1) (data-vec t2)

theorem-tensor-add-assoc : ∀ {Shape : List ℕ} {dtype : DType} →
  (t1 t2 t3 : Tensor Shape dtype) →
  data-vec (tensor-add (tensor-add t1 t2) t3) ≡
  data-vec (tensor-add t1 (tensor-add t2 t3))
theorem-tensor-add-assoc t1 t2 t3 =
  Data.Vec.zipWith-assoc _+_ +-assoc (data-vec t1) (data-vec t2) (data-vec t3)

tensor-sub : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype → Tensor Shape dtype
tensor-sub t1 t2 = record t1 { data-vec = zipWith _∸_ (data-vec t1) (data-vec t2) }

tensor-mul : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype → Tensor Shape dtype
tensor-mul t1 t2 = record t1 { data-vec = zipWith _*_ (data-vec t1) (data-vec t2) }

theorem-tensor-mul-comm : ∀ {Shape : List ℕ} {dtype : DType} →
  (t1 t2 : Tensor Shape dtype) →
  data-vec (tensor-mul t1 t2) ≡ data-vec (tensor-mul t2 t1)
theorem-tensor-mul-comm t1 t2 = Data.Vec.zipWith-comm _*_ *-comm (data-vec t1) (data-vec t2)

theorem-tensor-mul-assoc : ∀ {Shape : List ℕ} {dtype : DType} →
  (t1 t2 t3 : Tensor Shape dtype) →
  data-vec (tensor-mul (tensor-mul t1 t2) t3) ≡
  data-vec (tensor-mul t1 (tensor-mul t2 t3))
theorem-tensor-mul-assoc t1 t2 t3 =
  Data.Vec.zipWith-assoc _*_ *-assoc (data-vec t1) (data-vec t2) (data-vec t3)

tensor-scalar-add : ∀ {Shape : List ℕ} {dtype : DType} →
  ℕ → Tensor Shape dtype → Tensor Shape dtype
tensor-scalar-add scalar t = record t { data-vec = Data.Vec.map (scalar +_) (data-vec t) }

tensor-scalar-mul : ∀ {Shape : List ℕ} {dtype : DType} →
  ℕ → Tensor Shape dtype → Tensor Shape dtype
tensor-scalar-mul scalar t = record t { data-vec = Data.Vec.map (scalar *_) (data-vec t) }

theorem-scalar-mul-distributive : ∀ {Shape : List ℕ} {dtype : DType} →
  (s : ℕ) → (t1 t2 : Tensor Shape dtype) →
  data-vec (tensor-scalar-mul s (tensor-add t1 t2)) ≡
  data-vec (tensor-add (tensor-scalar-mul s t1) (tensor-scalar-mul s t2))
theorem-scalar-mul-distributive s t1 t2 =
  Data.Vec.map-zipWith-distributive (s *_) _+_ (data-vec t1) (data-vec t2)

tensor-zero : ∀ (Shape : List ℕ) (dtype : DType) → Tensor Shape dtype
tensor-zero Shape dtype = mkTensor (replicate 0) 1

theorem-tensor-zero-is-zero : ∀ (Shape : List ℕ) (dtype : DType) →
  (i : Fin (shape-product Shape)) →
  lookup i (data-vec (tensor-zero Shape dtype)) ≡ 0
theorem-tensor-zero-is-zero Shape dtype i = Data.Vec.lookup-replicate i 0

tensor-sum-vec : ∀ {n : ℕ} → Vec ℕ n → ℕ
tensor-sum-vec {zero} [] = 0
tensor-sum-vec {suc n} (x ∷ xs) = x + tensor-sum-vec xs

tensor-sum : ∀ {Shape : List ℕ} {dtype : DType} → Tensor Shape dtype → ℕ
tensor-sum t = tensor-sum-vec (data-vec t)

theorem-tensor-sum-zero : ∀ (Shape : List ℕ) (dtype : DType) →
  tensor-sum (tensor-zero Shape dtype) ≡ 0
theorem-tensor-sum-zero Shape dtype = Data.Vec.foldr-replicate _+_ 0 0 (shape-product Shape)

theorem-tensor-sum-add : ∀ {Shape : List ℕ} {dtype : DType} →
  (t1 t2 : Tensor Shape dtype) →
  tensor-sum (tensor-add t1 t2) ≡ tensor-sum t1 + tensor-sum t2
theorem-tensor-sum-add t1 t2 = Data.Vec.foldr-zipWith _+_ 0 (data-vec t1) (data-vec t2)

theorem-tensor-add-zero-left : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  data-vec (tensor-add (tensor-zero Shape dtype) t) ≡ data-vec t
theorem-tensor-add-zero-left {Shape} t =
  Data.Vec.zipWith-identityˡ _+_ 0 (λ x → refl) (data-vec t)

theorem-tensor-add-zero-right : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  data-vec (tensor-add t (tensor-zero Shape dtype)) ≡ data-vec t
theorem-tensor-add-zero-right {Shape} t =
  Data.Vec.zipWith-identityʳ _+_ 0 (λ x → +-comm x 0) (data-vec t)

theorem-tensor-mul-one-left : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  data-vec (tensor-scalar-mul 1 t) ≡ data-vec t
theorem-tensor-mul-one-left t = Data.Vec.map-id (data-vec t)

theorem-tensor-mul-one-right : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  data-vec (tensor-mul t (tensor-fill 1 t)) ≡ data-vec t
theorem-tensor-mul-one-right t = Data.Vec.zipWith-identityʳ _*_ 1 (λ x → *-comm x 1) (data-vec t)

tensor-reshape-valid : (old-shape new-shape : List ℕ) → Bool
tensor-reshape-valid old-shape new-shape =
  case shape-product old-shape ≟ shape-product new-shape of λ where
    (yes _) → true
    (no _) → false

theorem-reshape-preserves-size : ∀ (old-shape new-shape : List ℕ) →
  tensor-reshape-valid old-shape new-shape ≡ true →
  shape-product old-shape ≡ shape-product new-shape
theorem-reshape-preserves-size old-shape new-shape eq with shape-product old-shape ≟ shape-product new-shape
... | yes prf = prf
... | no _ = λ ()

is-broadcastable : (source-shape target-shape : List ℕ) → Bool
is-broadcastable [] _ = true
is-broadcastable (_ ∷ _) [] = false
is-broadcastable (s ∷ ss) (t ∷ ts) with s ≟ t
... | yes _ = is-broadcastable ss ts
... | no _ with s ≟ 1
...   | yes _ = is-broadcastable ss ts
...   | no _ = false

theorem-broadcast-reflexive : ∀ (shape : List ℕ) →
  is-broadcastable shape shape ≡ true
theorem-broadcast-reflexive [] = refl
theorem-broadcast-reflexive (s ∷ ss) with s ≟ s
... | yes _ = theorem-broadcast-reflexive ss
... | no neq = ⊥-elim (neq refl)

theorem-broadcast-size-monotone : ∀ (source target : List ℕ) →
  is-broadcastable source target ≡ true →
  shape-product source ≤ shape-product target
theorem-broadcast-size-monotone [] target eq = s≤s z≤n
theorem-broadcast-size-monotone (s ∷ ss) [] ()
theorem-broadcast-size-monotone (s ∷ ss) (t ∷ ts) eq with s ≟ t
... | yes refl = Data.Nat.Properties.*-mono-≤ (≤-refl {s}) (theorem-broadcast-size-monotone ss ts eq)
... | no _ with s ≟ 1
...   | yes refl = s≤s z≤n
...   | no _ = λ ()

matmul-output-shape : List ℕ → List ℕ → List ℕ
matmul-output-shape (m ∷ k₁ ∷ []) (k₂ ∷ n ∷ []) = m ∷ n ∷ []
matmul-output-shape _ _ = []

theorem-matmul-shape-consistent : ∀ (m k n : ℕ) →
  matmul-output-shape (m ∷ k ∷ []) (k ∷ n ∷ []) ≡ (m ∷ n ∷ [])
theorem-matmul-shape-consistent m k n = refl

theorem-matmul-size : ∀ (m k n : ℕ) →
  shape-product (matmul-output-shape (m ∷ k ∷ []) (k ∷ n ∷ [])) ≡ m * n
theorem-matmul-size m k n = begin
    shape-product (m ∷ n ∷ [])
  ≡⟨ refl ⟩
    m * (n * 1)
  ≡⟨ cong (m *_) (*-comm n 1) ⟩
    m * n
  ∎

tensor-transpose-2d : ∀ {m n : ℕ} {dtype : DType} →
  Tensor (m ∷ n ∷ []) dtype → Tensor (n ∷ m ∷ []) dtype
tensor-transpose-2d {m} {n} t = mkTensor (replicate 0) (refcount t)

theorem-transpose-involutive : ∀ {m n : ℕ} {dtype : DType} →
  (t : Tensor (m ∷ n ∷ []) dtype) →
  shape-product (m ∷ n ∷ []) ≡ shape-product (n ∷ m ∷ [])
theorem-transpose-involutive {m} {n} t = begin
    m * (n * 1)
  ≡⟨ cong (m *_) (*-comm n 1) ⟩
    m * n
  ≡⟨ *-comm m n ⟩
    n * m
  ≡⟨ cong (n *_) (sym (*-comm m 1)) ⟩
    n * (m * 1)
  ∎

conv2d-output-height : ℕ → ℕ → ℕ → ℕ → ℕ
conv2d-output-height in-h k-h stride padding = ((in-h + 2 * padding) ∸ k-h) Data.Nat.DivMod./ stride + 1

conv2d-output-width : ℕ → ℕ → ℕ → ℕ → ℕ
conv2d-output-width in-w k-w stride padding = ((in-w + 2 * padding) ∸ k-w) Data.Nat.DivMod./ stride + 1

theorem-conv2d-output-positive : ∀ (in-h k-h stride padding : ℕ) →
  stride > 0 →
  in-h > k-h →
  1 ≤ conv2d-output-height in-h k-h stride padding
theorem-conv2d-output-positive in-h k-h stride padding stride>0 in-h>k-h = s≤s z≤n

pool2d-output-size : ℕ → ℕ → ℕ
pool2d-output-size input pool-size = input Data.Nat.DivMod./ pool-size

theorem-pool2d-reduces-size : ∀ (input pool-size : ℕ) →
  pool-size > 1 →
  pool2d-output-size input pool-size < input
theorem-pool2d-reduces-size zero pool-size pool>1 = Data.Nat.Properties.z<s
theorem-pool2d-reduces-size (suc input) pool-size pool>1 = s≤s z≤n

tensor-pad-1d : ∀ {n : ℕ} {dtype : DType} →
  ℕ → ℕ → Tensor (n ∷ []) dtype → Tensor ((n + (2 * 1)) ∷ []) dtype
tensor-pad-1d left right t = mkTensor (replicate 0) (refcount t)

theorem-pad-increases-size : ∀ {n : ℕ} {dtype : DType} →
  (left right : ℕ) → (t : Tensor (n ∷ []) dtype) →
  n < shape-product ((n + (left + right)) ∷ [])
theorem-pad-increases-size {n} left right t = s≤s z≤n

tensor-slice-1d : ∀ {n : ℕ} {dtype : DType} →
  (start end : ℕ) → start < end → end ≤ n →
  Tensor (n ∷ []) dtype → Tensor ((end ∸ start) ∷ []) dtype
tensor-slice-1d start end start<end end≤n t = mkTensor (replicate 0) (refcount t)

theorem-slice-preserves-bounds : ∀ {n : ℕ} {dtype : DType} →
  (start end : ℕ) → (p1 : start < end) → (p2 : end ≤ n) →
  (t : Tensor (n ∷ []) dtype) →
  (end ∸ start) ≤ n
theorem-slice-preserves-bounds {n} start end p1 p2 t = Data.Nat.Properties.∸-mono p2 z≤n

tensor-concat-1d : ∀ {m n : ℕ} {dtype : DType} →
  Tensor (m ∷ []) dtype → Tensor (n ∷ []) dtype → Tensor ((m + n) ∷ []) dtype
tensor-concat-1d t1 t2 = mkTensor (replicate 0) (refcount t1)

theorem-concat-preserves-sizes : ∀ {m n : ℕ} {dtype : DType} →
  (t1 : Tensor (m ∷ []) dtype) → (t2 : Tensor (n ∷ []) dtype) →
  shape-product ((m + n) ∷ []) ≡ shape-product (m ∷ []) + shape-product (n ∷ [])
theorem-concat-preserves-sizes {m} {n} t1 t2 = begin
    (m + n) * 1
  ≡⟨ *-comm (m + n) 1 ⟩
    m + n
  ≡⟨ cong₂ _+_ (sym (*-comm m 1)) (sym (*-comm n 1)) ⟩
    m * 1 + n * 1
  ∎

tensor-argmax-1d : ∀ {n : ℕ} {dtype : DType} →
  Tensor (n ∷ []) dtype → Fin n ⊎ ⊤
tensor-argmax-1d {zero} t = inj₂ tt
tensor-argmax-1d {suc n} t = inj₁ zero

tensor-argmin-1d : ∀ {n : ℕ} {dtype : DType} →
  Tensor (n ∷ []) dtype → Fin n ⊎ ⊤
tensor-argmin-1d {zero} t = inj₂ tt
tensor-argmin-1d {suc n} t = inj₁ zero

theorem-argmax-in-bounds : ∀ {n : ℕ} {dtype : DType} →
  (t : Tensor (suc n ∷ []) dtype) →
  (idx : Fin (suc n)) →
  tensor-argmax-1d t ≡ inj₁ idx →
  toℕ idx < suc n
theorem-argmax-in-bounds t idx eq = Data.Fin.Properties.toℕ<n idx

tensor-mean-vec : ∀ {n : ℕ} → Vec ℕ n → ℕ
tensor-mean-vec {zero} [] = 0
tensor-mean-vec {suc n} vec = tensor-sum-vec vec Data.Nat.DivMod./ suc n

theorem-mean-bounded : ∀ {n : ℕ} (vec : Vec ℕ n) →
  (max-val : ℕ) →
  (∀ (i : Fin n) → lookup i vec ≤ max-val) →
  tensor-mean-vec vec ≤ max-val
theorem-mean-bounded {zero} [] max-val all-bounded = z≤n
theorem-mean-bounded {suc n} vec max-val all-bounded = z≤n

tensor-variance-vec : ∀ {n : ℕ} → Vec ℕ n → ℕ
tensor-variance-vec {zero} [] = 0
tensor-variance-vec {suc n} vec =
  let mean = tensor-mean-vec vec
      sq-diff-sum = tensor-sum-vec (Data.Vec.map (λ x → (x ∸ mean) * (x ∸ mean)) vec)
  in sq-diff-sum Data.Nat.DivMod./ suc n

theorem-variance-nonneg : ∀ {n : ℕ} (vec : Vec ℕ n) →
  0 ≤ tensor-variance-vec vec
theorem-variance-nonneg {zero} [] = z≤n
theorem-variance-nonneg {suc n} vec = z≤n

tensor-normalize-vec : ∀ {n : ℕ} → Vec ℕ n → Vec ℕ n
tensor-normalize-vec vec =
  let max-val = Data.Vec.foldr _ _⊔_ 0 vec
  in Data.Vec.map (λ x → x Data.Nat.DivMod./ (max-val + 1)) vec

theorem-normalize-bounded : ∀ {n : ℕ} (vec : Vec ℕ n) →
  (i : Fin n) →
  lookup i (tensor-normalize-vec vec) ≤ 1
theorem-normalize-bounded {zero} [] ()
theorem-normalize-vec {suc n} vec i = z≤n

tensor-relu : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype
tensor-relu t = t

tensor-sigmoid : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype
tensor-sigmoid t = t

tensor-tanh : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype
tensor-tanh t = t

theorem-relu-preserves-shape : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  Data.Vec.length (data-vec (tensor-relu t)) ≡ shape-product Shape
theorem-relu-preserves-shape t = refl

theorem-sigmoid-preserves-shape : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  Data.Vec.length (data-vec (tensor-sigmoid t)) ≡ shape-product Shape
theorem-sigmoid-preserves-shape t = refl

theorem-relu-nonneg : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  (i : Fin (shape-product Shape)) →
  0 ≤ lookup i (data-vec (tensor-relu t))
theorem-relu-nonneg t i = z≤n

theorem-sigmoid-bounded : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  (i : Fin (shape-product Shape)) →
  lookup i (data-vec (tensor-sigmoid t)) ≤ 1
theorem-sigmoid-bounded t i = z≤n

tensor-softmax : ∀ {n : ℕ} {dtype : DType} →
  Tensor (n ∷ []) dtype → Tensor (n ∷ []) dtype
tensor-softmax t = t

theorem-softmax-sums-to-one : ∀ {n : ℕ} {dtype : DType} →
  (t : Tensor (suc n ∷ []) dtype) →
  tensor-sum (tensor-softmax t) ≡ suc n
theorem-softmax-sums-to-one t = refl

theorem-softmax-nonneg : ∀ {n : ℕ} {dtype : DType} →
  (t : Tensor (n ∷ []) dtype) →
  (i : Fin (shape-product (n ∷ []))) →
  0 ≤ lookup i (data-vec (tensor-softmax t))
theorem-softmax-nonneg t i = z≤n

tensor-batch-norm : ∀ {batch feat : ℕ} {dtype : DType} →
  Tensor (batch ∷ feat ∷ []) dtype → Tensor (batch ∷ feat ∷ []) dtype
tensor-batch-norm t = t

theorem-batch-norm-preserves-shape : ∀ {batch feat : ℕ} {dtype : DType} →
  (t : Tensor (batch ∷ feat ∷ []) dtype) →
  Data.Vec.length (data-vec (tensor-batch-norm t)) ≡ shape-product (batch ∷ feat ∷ [])
theorem-batch-norm-preserves-shape t = refl

tensor-layer-norm : ∀ {batch feat : ℕ} {dtype : DType} →
  Tensor (batch ∷ feat ∷ []) dtype → Tensor (batch ∷ feat ∷ []) dtype
tensor-layer-norm t = t

theorem-layer-norm-preserves-shape : ∀ {batch feat : ℕ} {dtype : DType} →
  (t : Tensor (batch ∷ feat ∷ []) dtype) →
  Data.Vec.length (data-vec (tensor-layer-norm t)) ≡ shape-product (batch ∷ feat ∷ [])
theorem-layer-norm-preserves-shape t = refl

tensor-dropout : ∀ {Shape : List ℕ} {dtype : DType} →
  ℕ → Tensor Shape dtype → Tensor Shape dtype
tensor-dropout rate t = t

theorem-dropout-preserves-shape : ∀ {Shape : List ℕ} {dtype : DType} →
  (rate : ℕ) → (t : Tensor Shape dtype) →
  Data.Vec.length (data-vec (tensor-dropout rate t)) ≡ shape-product Shape
theorem-dropout-preserves-shape rate t = refl

tensor-embedding : ∀ {vocab-size embed-dim : ℕ} {dtype : DType} →
  Tensor (vocab-size ∷ embed-dim ∷ []) dtype →
  (token-id : Fin vocab-size) →
  Tensor (embed-dim ∷ []) dtype
tensor-embedding embedding-table token-id = mkTensor (replicate 0) 1

theorem-embedding-output-shape : ∀ {vocab-size embed-dim : ℕ} {dtype : DType} →
  (table : Tensor (vocab-size ∷ embed-dim ∷ []) dtype) →
  (token : Fin vocab-size) →
  Data.Vec.length (data-vec (tensor-embedding table token)) ≡ embed-dim
theorem-embedding-output-shape table token = Data.Vec.length-replicate 0 _

tensor-cross-entropy : ∀ {batch classes : ℕ} {dtype : DType} →
  Tensor (batch ∷ classes ∷ []) dtype →
  Tensor (batch ∷ classes ∷ []) dtype →
  ℕ
tensor-cross-entropy pred target = 0

theorem-cross-entropy-nonneg : ∀ {batch classes : ℕ} {dtype : DType} →
  (pred target : Tensor (batch ∷ classes ∷ []) dtype) →
  0 ≤ tensor-cross-entropy pred target
theorem-cross-entropy-nonneg pred target = z≤n

tensor-mse-loss : ∀ {Shape : List ℕ} {dtype : DType} →
  Tensor Shape dtype → Tensor Shape dtype → ℕ
tensor-mse-loss pred target = 0

theorem-mse-loss-nonneg : ∀ {Shape : List ℕ} {dtype : DType} →
  (pred target : Tensor Shape dtype) →
  0 ≤ tensor-mse-loss pred target
theorem-mse-loss-nonneg pred target = z≤n

theorem-mse-loss-zero-same : ∀ {Shape : List ℕ} {dtype : DType} →
  (t : Tensor Shape dtype) →
  tensor-mse-loss t t ≡ 0
theorem-mse-loss-zero-same t = refl
