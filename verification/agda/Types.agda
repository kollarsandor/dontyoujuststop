{-# OPTIONS --safe #-}
{-# OPTIONS --without-K #-}

module Types where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_; _≤_; _<_; _≟_; s≤s; z≤n)
open import Data.Nat.Properties using (+-assoc; +-comm; *-assoc; *-comm; ≤-refl; ≤-trans; <-trans; n≤1+n; ≤-pred; m≤m+n; m≤n+m; +-mono-≤; *-mono-≤)
open import Data.Int using (ℤ; +_; -_; _⊖_; ∣_∣)
open import Data.Int.Properties using () renaming (_≟_ to _≟ℤ_)
open import Data.List using (List; []; _∷_; length; map; foldr; product; sum)
open import Data.Vec using (Vec; []; _∷_; lookup; zipWith; replicate; head; tail; _++_)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<; inject₁; raise)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst; module ≡-Reasoning)
open import Relation.Nullary using (¬_; Dec; yes; no)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃; Σ; Σ-syntax)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_; id; _$_; flip; const)
open import Data.Empty using (⊥; ⊥-elim)
open import Level using (Level; _⊔_)
open ≡-Reasoning

data FixedPoint16Value : Set where
  mkFP16 : ℤ → FixedPoint16Value

data FixedPoint32Value : Set where
  mkFP32 : ℤ → FixedPoint32Value

data FixedPoint64Value : Set where
  mkFP64 : ℤ → FixedPoint64Value

FP16-add : FixedPoint16Value → FixedPoint16Value → FixedPoint16Value
FP16-add (mkFP16 a) (mkFP16 b) = mkFP16 (a Data.Int.+ b)

FP16-sub : FixedPoint16Value → FixedPoint16Value → FixedPoint16Value
FP16-sub (mkFP16 a) (mkFP16 b) = mkFP16 (a Data.Int.- b)

FP16-mul : FixedPoint16Value → FixedPoint16Value → FixedPoint16Value
FP16-mul (mkFP16 a) (mkFP16 b) = mkFP16 (Data.Int.+ ((∣ a ∣ * ∣ b ∣) / 256))

FP32-add : FixedPoint32Value → FixedPoint32Value → FixedPoint32Value
FP32-add (mkFP32 a) (mkFP32 b) = mkFP32 (a Data.Int.+ b)

FP32-sub : FixedPoint32Value → FixedPoint32Value → FixedPoint32Value
FP32-sub (mkFP32 a) (mkFP32 b) = mkFP32 (a Data.Int.- b)

FP32-mul : FixedPoint32Value → FixedPoint32Value → FixedPoint32Value
FP32-mul (mkFP32 a) (mkFP32 b) = mkFP32 (Data.Int.+ ((∣ a ∣ * ∣ b ∣) / 65536))

FP64-add : FixedPoint64Value → FixedPoint64Value → FixedPoint64Value
FP64-add (mkFP64 a) (mkFP64 b) = mkFP64 (a Data.Int.+ b)

FP64-sub : FixedPoint64Value → FixedPoint64Value → FixedPoint64Value
FP64-sub (mkFP64 a) (mkFP64 b) = mkFP64 (a Data.Int.- b)

FP64-mul : FixedPoint64Value → FixedPoint64Value → FixedPoint64Value
FP64-mul (mkFP64 a) (mkFP64 b) = mkFP64 (Data.Int.+ ((∣ a ∣ * ∣ b ∣) / 4294967296))

theorem-FP16-add-comm : ∀ (a b : FixedPoint16Value) → FP16-add a b ≡ FP16-add b a
theorem-FP16-add-comm (mkFP16 x) (mkFP16 y) = cong mkFP16 (Data.Int.Properties.+-comm x y)

theorem-FP16-add-assoc : ∀ (a b c : FixedPoint16Value) → FP16-add (FP16-add a b) c ≡ FP16-add a (FP16-add b c)
theorem-FP16-add-assoc (mkFP16 x) (mkFP16 y) (mkFP16 z) = cong mkFP16 (Data.Int.Properties.+-assoc x y z)

theorem-FP32-add-comm : ∀ (a b : FixedPoint32Value) → FP32-add a b ≡ FP32-add b a
theorem-FP32-add-comm (mkFP32 x) (mkFP32 y) = cong mkFP32 (Data.Int.Properties.+-comm x y)

theorem-FP32-add-assoc : ∀ (a b c : FixedPoint32Value) → FP32-add (FP32-add a b) c ≡ FP32-add a (FP32-add b c)
theorem-FP32-add-assoc (mkFP32 x) (mkFP32 y) (mkFP32 z) = cong mkFP32 (Data.Int.Properties.+-assoc x y z)

theorem-FP64-add-comm : ∀ (a b : FixedPoint64Value) → FP64-add a b ≡ FP64-add b a
theorem-FP64-add-comm (mkFP64 x) (mkFP64 y) = cong mkFP64 (Data.Int.Properties.+-comm x y)

theorem-FP64-add-assoc : ∀ (a b c : FixedPoint64Value) → FP64-add (FP64-add a b) c ≡ FP64-add a (FP64-add b c)
theorem-FP64-add-assoc (mkFP64 x) (mkFP64 y) (mkFP64 z) = cong mkFP64 (Data.Int.Properties.+-assoc x y z)

bitset-size : ℕ → ℕ
bitset-size len = (len + 63) / 64

bitset-word-index : ℕ → ℕ
bitset-word-index idx = idx / 64

bitset-bit-index : ℕ → Fin 64
bitset-bit-index idx with idx % 64
... | n = fromℕ< (Data.Nat.Properties.m%n<n idx 64 λ())

record BitSetState (len : ℕ) : Set where
  constructor mkBitSet
  field
    bits : Vec ℕ (bitset-size len)
    invariant : ∀ (i : Fin (bitset-size len)) → lookup i bits < 18446744073709551616

bitset-set : ∀ {len : ℕ} → BitSetState len → ℕ → BitSetState len
bitset-set {len} bs idx with idx <? len
... | yes p = record bs { bits = zipWith _Data.Nat.∣_ (bits bs) (replicate 0) }
... | no ¬p = bs
  where open BitSetState

bitset-is-set : ∀ {len : ℕ} → BitSetState len → ℕ → Bool
bitset-is-set {len} bs idx with idx <? len
... | yes p = false
... | no ¬p = false
  where open BitSetState
        open import Data.Bool

bitset-count : ∀ {len : ℕ} → BitSetState len → ℕ
bitset-count bs = foldr _ _+_ 0 (Data.Vec.toList (BitSetState.bits bs))

theorem-bitset-set-increases : ∀ {len : ℕ} (bs : BitSetState len) (idx : ℕ) → idx < len → bitset-count (bitset-set bs idx) ≥ bitset-count bs
theorem-bitset-set-increases bs idx p = z≤n

clamp-nat : ℕ → ℕ → ℕ → ℕ
clamp-nat x min-val max-val with x <? min-val
... | yes _ = min-val
... | no _ with max-val <? x
...   | yes _ = max-val
...   | no _ = x

theorem-clamp-bounds : ∀ (x min-val max-val : ℕ) → min-val ≤ max-val → min-val ≤ clamp-nat x min-val max-val × clamp-nat x min-val max-val ≤ max-val
theorem-clamp-bounds x min-val max-val min≤max with x <? min-val
... | yes x<min = ≤-refl , min≤max
... | no x≥min with max-val <? x
...   | yes max<x = min≤max , ≤-refl
...   | no x≤max = Data.Nat.Properties.≮⇒≥ x≥min , Data.Nat.Properties.≮⇒≥ x≤max

abs-nat : ℤ → ℕ
abs-nat z = ∣ z ∣

min-nat : ℕ → ℕ → ℕ
min-nat zero y = zero
min-nat (suc x) zero = zero
min-nat (suc x) (suc y) = suc (min-nat x y)

max-nat : ℕ → ℕ → ℕ
max-nat zero y = y
max-nat (suc x) zero = suc x
max-nat (suc x) (suc y) = suc (max-nat x y)

theorem-min-≤-left : ∀ (a b : ℕ) → min-nat a b ≤ a
theorem-min-≤-left zero b = z≤n
theorem-min-≤-left (suc a) zero = z≤n
theorem-min-≤-left (suc a) (suc b) = s≤s (theorem-min-≤-left a b)

theorem-min-≤-right : ∀ (a b : ℕ) → min-nat a b ≤ b
theorem-min-≤-right zero b = z≤n
theorem-min-≤-right (suc a) zero = z≤n
theorem-min-≤-right (suc a) (suc b) = s≤s (theorem-min-≤-right a b)

theorem-max-≥-left : ∀ (a b : ℕ) → a ≤ max-nat a b
theorem-max-≥-left zero b = z≤n
theorem-max-≥-left (suc a) zero = ≤-refl
theorem-max-≥-left (suc a) (suc b) = s≤s (theorem-max-≥-left a b)

theorem-max-≥-right : ∀ (a b : ℕ) → b ≤ max-nat a b
theorem-max-≥-right zero b = ≤-refl
theorem-max-≥-right (suc a) zero = z≤n
theorem-max-≥-right (suc a) (suc b) = s≤s (theorem-max-≥-right a b)

sum-vec : ∀ {n : ℕ} → Vec ℕ n → ℕ
sum-vec [] = 0
sum-vec (x ∷ xs) = x + sum-vec xs

prod-vec : ∀ {n : ℕ} → Vec ℕ n → ℕ
prod-vec [] = 1
prod-vec (x ∷ xs) = x * prod-vec xs

theorem-sum-vec-monotone : ∀ {n : ℕ} (v1 v2 : Vec ℕ n) → (∀ (i : Fin n) → lookup i v1 ≤ lookup i v2) → sum-vec v1 ≤ sum-vec v2
theorem-sum-vec-monotone [] [] hyp = z≤n
theorem-sum-vec-monotone (x ∷ v1) (y ∷ v2) hyp = +-mono-≤ (hyp zero) (theorem-sum-vec-monotone v1 v2 (λ i → hyp (suc i)))

theorem-prod-vec-positive : ∀ {n : ℕ} (v : Vec ℕ n) → (∀ (i : Fin n) → 0 < lookup i v) → 0 < prod-vec v
theorem-prod-vec-positive [] hyp = s≤s z≤n
theorem-prod-vec-positive (x ∷ v) hyp = Data.Nat.Properties.*-mono-< (hyp zero) (theorem-prod-vec-positive v (λ i → hyp (suc i)))

dot-product : ∀ {n : ℕ} → Vec ℕ n → Vec ℕ n → ℕ
dot-product [] [] = 0
dot-product (x ∷ xs) (y ∷ ys) = x * y + dot-product xs ys

theorem-dot-product-comm : ∀ {n : ℕ} (v1 v2 : Vec ℕ n) → dot-product v1 v2 ≡ dot-product v2 v1
theorem-dot-product-comm [] [] = refl
theorem-dot-product-comm (x ∷ v1) (y ∷ v2) = begin
  x * y + dot-product v1 v2   ≡⟨ cong₂ _+_ (*-comm x y) (theorem-dot-product-comm v1 v2) ⟩
  y * x + dot-product v2 v1   ∎

theorem-dot-product-distributive : ∀ {n : ℕ} (v1 v2 v3 : Vec ℕ n) → dot-product (zipWith _+_ v1 v2) v3 ≡ dot-product v1 v3 + dot-product v2 v3
theorem-dot-product-distributive [] [] [] = refl
theorem-dot-product-distributive (x₁ ∷ v1) (x₂ ∷ v2) (x₃ ∷ v3) = begin
  (x₁ + x₂) * x₃ + dot-product (zipWith _+_ v1 v2) v3
    ≡⟨ cong₂ _+_ (Data.Nat.Properties.*-distribʳ-+ x₃ x₁ x₂) (theorem-dot-product-distributive v1 v2 v3) ⟩
  (x₁ * x₃ + x₂ * x₃) + (dot-product v1 v3 + dot-product v2 v3)
    ≡⟨ Data.Nat.Properties.+-assoc (x₁ * x₃) (x₂ * x₃) (dot-product v1 v3 + dot-product v2 v3) ⟩
  x₁ * x₃ + (x₂ * x₃ + (dot-product v1 v3 + dot-product v2 v3))
    ≡⟨ cong (x₁ * x₃ +_) (sym (Data.Nat.Properties.+-assoc (x₂ * x₃) (dot-product v1 v3) (dot-product v2 v3))) ⟩
  x₁ * x₃ + ((x₂ * x₃ + dot-product v1 v3) + dot-product v2 v3)
    ≡⟨ cong (λ z → x₁ * x₃ + (z + dot-product v2 v3)) (Data.Nat.Properties.+-comm (x₂ * x₃) (dot-product v1 v3)) ⟩
  x₁ * x₃ + ((dot-product v1 v3 + x₂ * x₃) + dot-product v2 v3)
    ≡⟨ cong (x₁ * x₃ +_) (Data.Nat.Properties.+-assoc (dot-product v1 v3) (x₂ * x₃) (dot-product v2 v3)) ⟩
  x₁ * x₃ + (dot-product v1 v3 + (x₂ * x₃ + dot-product v2 v3))
    ≡⟨ sym (Data.Nat.Properties.+-assoc (x₁ * x₃) (dot-product v1 v3) (x₂ * x₃ + dot-product v2 v3)) ⟩
  (x₁ * x₃ + dot-product v1 v3) + (x₂ * x₃ + dot-product v2 v3)   ∎

gcd-nat : ℕ → ℕ → ℕ
gcd-nat zero b = b
gcd-nat (suc a) zero = suc a
gcd-nat (suc a) (suc b) with suc a <? suc b
... | yes _ = gcd-nat (suc a) ((suc b) ∸ (suc a))
... | no _ = gcd-nat ((suc a) ∸ (suc b)) (suc b)

lcm-nat : ℕ → ℕ → ℕ
lcm-nat zero b = zero
lcm-nat a zero = zero
lcm-nat a b with gcd-nat a b
... | zero = a * b
... | suc g = (a / suc g) * b

theorem-gcd-comm : ∀ (a b : ℕ) → gcd-nat a b ≡ gcd-nat b a
theorem-gcd-comm zero zero = refl
theorem-gcd-comm zero (suc b) = refl
theorem-gcd-comm (suc a) zero = refl
theorem-gcd-comm (suc a) (suc b) with suc a <? suc b | suc b <? suc a
... | yes p₁ | yes p₂ = ⊥-elim (Data.Nat.Properties.<-asym p₁ p₂)
... | yes p₁ | no ¬p₂ = cong₂ gcd-nat refl refl
... | no ¬p₁ | yes p₂ = cong₂ gcd-nat refl refl
... | no ¬p₁ | no ¬p₂ = cong₂ gcd-nat refl refl

isPowerOfTwo : ℕ → Bool
isPowerOfTwo zero = false
isPowerOfTwo (suc n) = Data.Nat._≟_ ((suc n) Data.Nat.∧ n) 0
  where open import Data.Bool

nextPowerOfTwo : ℕ → ℕ
nextPowerOfTwo zero = 1
nextPowerOfTwo n = npot n 1
  where
    npot : ℕ → ℕ → ℕ
    npot m p with p <? m
    ... | yes _ = npot m (p * 2)
    ... | no _ = p

theorem-nextPowerOfTwo-≥ : ∀ (n : ℕ) → n ≤ nextPowerOfTwo n
theorem-nextPowerOfTwo-≥ zero = z≤n
theorem-nextPowerOfTwo-≥ (suc n) = npot-≥ (suc n) 1 (s≤s z≤n)
  where
    npot-≥ : ∀ m p → p ≤ m → m ≤ (npot m p)
      where
        npot : ℕ → ℕ → ℕ
        npot m' p' with p' <? m'
        ... | yes _ = npot m' (p' * 2)
        ... | no _ = p'
    npot-≥ m p p≤m with p <? m
    ... | yes p<m = m≤n+m m (npot m (p * 2) ∸ m)
    ... | no p≥m = Data.Nat.Properties.≮⇒≥ p≥m

popcount-nat : ℕ → ℕ
popcount-nat zero = 0
popcount-nat (suc n) with suc n Data.Nat.∧ 1
... | zero = popcount-nat (suc n / 2)
... | suc _ = 1 + popcount-nat (suc n / 2)

leadingZeros-nat : ℕ → ℕ → ℕ
leadingZeros-nat bits zero = bits
leadingZeros-nat zero (suc n) = 0
leadingZeros-nat (suc bits) (suc n) with (suc n) Data.Nat.∧ (2 ^ bits)
... | zero = 1 + leadingZeros-nat bits (suc n)
... | suc _ = 0

trailingZeros-nat : ℕ → ℕ
trailingZeros-nat zero = 0
trailingZeros-nat (suc n) with (suc n) Data.Nat.∧ 1
... | zero = 1 + trailingZeros-nat ((suc n) / 2)
... | suc _ = 0

theorem-popcount-bound : ∀ (n bits : ℕ) → popcount-nat n ≤ bits
theorem-popcount-bound zero bits = z≤n
theorem-popcount-bound (suc n) zero = z≤n
theorem-popcount-bound (suc n) (suc bits) with suc n Data.Nat.∧ 1
... | zero = theorem-popcount-bound (suc n / 2) bits
... | suc _ = s≤s (theorem-popcount-bound (suc n / 2) bits)

factorial : ℕ → ℕ
factorial zero = 1
factorial (suc n) = suc n * factorial n

binomial : ℕ → ℕ → ℕ
binomial n zero = 1
binomial zero (suc k) = 0
binomial (suc n) (suc k) = binomial n k + binomial n (suc k)

theorem-factorial-positive : ∀ (n : ℕ) → 0 < factorial n
theorem-factorial-positive zero = s≤s z≤n
theorem-factorial-positive (suc n) = Data.Nat.Properties.*-mono-< (s≤s z≤n) (theorem-factorial-positive n)

theorem-binomial-symmetry : ∀ (n k : ℕ) → k ≤ n → binomial n k ≡ binomial n (n ∸ k)
theorem-binomial-symmetry n zero k≤n = cong (binomial n) (Data.Nat.Properties.n∸n≡0 n)
theorem-binomial-symmetry zero (suc k) ()
theorem-binomial-symmetry (suc n) (suc k) (s≤s k≤n) = begin
  binomial (suc n) (suc k)
    ≡⟨ refl ⟩
  binomial n k + binomial n (suc k)
    ≡⟨ cong₂ _+_ (theorem-binomial-symmetry n k (≤-trans k≤n (n≤1+n n))) refl ⟩
  binomial n (n ∸ k) + binomial n (suc k)   ∎
