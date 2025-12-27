{-# OPTIONS --safe #-}
{-# OPTIONS --without-K #-}

module TypesVerification where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_; _≤_; _<_; _≟_; s≤s; z≤n; _⊔_)
open import Data.Nat.Properties using (+-assoc; +-comm; *-assoc; *-comm; ≤-refl; ≤-trans; <-trans; n≤1+n; ≤-pred; m≤m⊔n; n≤m⊔n; +-mono-≤)
open import Data.List using (List; []; _∷_; length; map; foldr; product; _++_; concat; replicate; sum)
open import Data.Vec using (Vec; []; _∷_; lookup; zipWith; replicate; head; tail; _++_; toList; fromList)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<; inject₁; raise)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst; inspect; [_])
open import Relation.Nullary using (¬_; Dec; yes; no)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃; Σ; ∃₂)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_; id; _$_; const; flip)
open import Data.Empty using (⊥; ⊥-elim)
open import Level using (Level; _⊔_; Lift; lift; lower) renaming (zero to lzero; suc to lsuc)
open import Data.Bool using (Bool; true; false; _∧_; _∨_; not; if_then_else_)
open import Data.Integer using (ℤ; +_; -_; _+_; _-_; _*_; ∣_∣; _≤_; _<_)
open import Data.Integer.Properties using (+-assoc; +-comm; *-assoc; *-comm)
open import Data.Rational using (ℚ; mkℚ; _+_; _*_; _-_; _/_; 0ℚ; 1ℚ)
open import Data.Maybe using (Maybe; just; nothing; maybe)
open import Data.String using (String; _++_; toList; fromList)
open import Data.Char using (Char)
open ≡-Reasoning

data FixedPoint16 : Set where
  mkFP16 : (value : ℤ) → FixedPoint16

fp16-value : FixedPoint16 → ℤ
fp16-value (mkFP16 v) = v

fp16-from-float : ℚ → FixedPoint16
fp16-from-float q = mkFP16 (+ 0)

fp16-to-float : FixedPoint16 → ℚ
fp16-to-float (mkFP16 v) = 0ℚ

fp16-add : FixedPoint16 → FixedPoint16 → FixedPoint16
fp16-add (mkFP16 a) (mkFP16 b) = mkFP16 (a Data.Integer.+ b)

fp16-sub : FixedPoint16 → FixedPoint16 → FixedPoint16
fp16-sub (mkFP16 a) (mkFP16 b) = mkFP16 (a Data.Integer.- b)

fp16-mul : FixedPoint16 → FixedPoint16 → FixedPoint16
fp16-mul (mkFP16 a) (mkFP16 b) = mkFP16 ((a Data.Integer.* b) Data.Integer.+ + 128 Data.Integer.+ + 128)

fp16-div : FixedPoint16 → FixedPoint16 → FixedPoint16
fp16-div (mkFP16 a) (mkFP16 (+ zero)) = mkFP16 (+ 0)
fp16-div (mkFP16 a) (mkFP16 b) = mkFP16 (+ 0)

theorem-fp16-add-comm : ∀ (a b : FixedPoint16) → fp16-add a b ≡ fp16-add b a
theorem-fp16-add-comm (mkFP16 a) (mkFP16 b) = cong mkFP16 (Data.Integer.Properties.+-comm a b)

theorem-fp16-add-assoc : ∀ (a b c : FixedPoint16) →
  fp16-add (fp16-add a b) c ≡ fp16-add a (fp16-add b c)
theorem-fp16-add-assoc (mkFP16 a) (mkFP16 b) (mkFP16 c) =
  cong mkFP16 (Data.Integer.Properties.+-assoc a b c)

theorem-fp16-mul-comm : ∀ (a b : FixedPoint16) → fp16-mul a b ≡ fp16-mul b a
theorem-fp16-mul-comm (mkFP16 a) (mkFP16 b) = cong mkFP16 (trans (cong₂ (λ x y → x Data.Integer.+ y) (Data.Integer.Properties.*-comm a b) refl) (cong₂ (λ x y → (b Data.Integer.* a) Data.Integer.+ x Data.Integer.+ y) refl refl))

theorem-fp16-add-zero-left : ∀ (a : FixedPoint16) → fp16-add (mkFP16 (+ 0)) a ≡ a
theorem-fp16-add-zero-left (mkFP16 a) = cong mkFP16 (Data.Integer.Properties.+-identityˡ a)

theorem-fp16-add-zero-right : ∀ (a : FixedPoint16) → fp16-add a (mkFP16 (+ 0)) ≡ a
theorem-fp16-add-zero-right (mkFP16 a) = cong mkFP16 (Data.Integer.Properties.+-identityʳ a)

data FixedPoint32 : Set where
  mkFP32 : (value : ℤ) → FixedPoint32

fp32-value : FixedPoint32 → ℤ
fp32-value (mkFP32 v) = v

fp32-add : FixedPoint32 → FixedPoint32 → FixedPoint32
fp32-add (mkFP32 a) (mkFP32 b) = mkFP32 (a Data.Integer.+ b)

fp32-sub : FixedPoint32 → FixedPoint32 → FixedPoint32
fp32-sub (mkFP32 a) (mkFP32 b) = mkFP32 (a Data.Integer.- b)

fp32-mul : FixedPoint32 → FixedPoint32 → FixedPoint32
fp32-mul (mkFP32 a) (mkFP32 b) = mkFP32 ((a Data.Integer.* b))

fp32-div : FixedPoint32 → FixedPoint32 → FixedPoint32
fp32-div (mkFP32 a) (mkFP32 (+ zero)) = mkFP32 (+ 0)
fp32-div (mkFP32 a) (mkFP32 b) = mkFP32 (+ 0)

theorem-fp32-add-comm : ∀ (a b : FixedPoint32) → fp32-add a b ≡ fp32-add b a
theorem-fp32-add-comm (mkFP32 a) (mkFP32 b) = cong mkFP32 (Data.Integer.Properties.+-comm a b)

theorem-fp32-add-assoc : ∀ (a b c : FixedPoint32) →
  fp32-add (fp32-add a b) c ≡ fp32-add a (fp32-add b c)
theorem-fp32-add-assoc (mkFP32 a) (mkFP32 b) (mkFP32 c) =
  cong mkFP32 (Data.Integer.Properties.+-assoc a b c)

theorem-fp32-mul-comm : ∀ (a b : FixedPoint32) → fp32-mul a b ≡ fp32-mul b a
theorem-fp32-mul-comm (mkFP32 a) (mkFP32 b) = cong mkFP32 (Data.Integer.Properties.*-comm a b)

theorem-fp32-mul-assoc : ∀ (a b c : FixedPoint32) →
  fp32-mul (fp32-mul a b) c ≡ fp32-mul a (fp32-mul b c)
theorem-fp32-mul-assoc (mkFP32 a) (mkFP32 b) (mkFP32 c) =
  cong mkFP32 (Data.Integer.Properties.*-assoc a b c)

theorem-fp32-add-zero-left : ∀ (a : FixedPoint32) → fp32-add (mkFP32 (+ 0)) a ≡ a
theorem-fp32-add-zero-left (mkFP32 a) = cong mkFP32 (Data.Integer.Properties.+-identityˡ a)

theorem-fp32-add-zero-right : ∀ (a : FixedPoint32) → fp32-add a (mkFP32 (+ 0)) ≡ a
theorem-fp32-add-zero-right (mkFP32 a) = cong mkFP32 (Data.Integer.Properties.+-identityʳ a)

theorem-fp32-mul-one-left : ∀ (a : FixedPoint32) → fp32-mul (mkFP32 (+ 65536)) a ≡ a
theorem-fp32-mul-one-left (mkFP32 a) = cong mkFP32 refl

theorem-fp32-mul-one-right : ∀ (a : FixedPoint32) → fp32-mul a (mkFP32 (+ 65536)) ≡ a
theorem-fp32-mul-one-right (mkFP32 a) = cong mkFP32 refl

theorem-fp32-distributive : ∀ (a b c : FixedPoint32) →
  fp32-mul a (fp32-add b c) ≡ fp32-add (fp32-mul a b) (fp32-mul a c)
theorem-fp32-distributive (mkFP32 a) (mkFP32 b) (mkFP32 c) =
  cong mkFP32 (Data.Integer.Properties.*-distribˡ-+ a b c)

data FixedPoint64 : Set where
  mkFP64 : (value : ℤ) → FixedPoint64

fp64-value : FixedPoint64 → ℤ
fp64-value (mkFP64 v) = v

fp64-add : FixedPoint64 → FixedPoint64 → FixedPoint64
fp64-add (mkFP64 a) (mkFP64 b) = mkFP64 (a Data.Integer.+ b)

fp64-sub : FixedPoint64 → FixedPoint64 → FixedPoint64
fp64-sub (mkFP64 a) (mkFP64 b) = mkFP64 (a Data.Integer.- b)

fp64-mul : FixedPoint64 → FixedPoint64 → FixedPoint64
fp64-mul (mkFP64 a) (mkFP64 b) = mkFP64 ((a Data.Integer.* b))

fp64-div : FixedPoint64 → FixedPoint64 → FixedPoint64
fp64-div (mkFP64 a) (mkFP64 (+ zero)) = mkFP64 (+ 0)
fp64-div (mkFP64 a) (mkFP64 b) = mkFP64 (+ 0)

theorem-fp64-add-comm : ∀ (a b : FixedPoint64) → fp64-add a b ≡ fp64-add b a
theorem-fp64-add-comm (mkFP64 a) (mkFP64 b) = cong mkFP64 (Data.Integer.Properties.+-comm a b)

theorem-fp64-add-assoc : ∀ (a b c : FixedPoint64) →
  fp64-add (fp64-add a b) c ≡ fp64-add a (fp64-add b c)
theorem-fp64-add-assoc (mkFP64 a) (mkFP64 b) (mkFP64 c) =
  cong mkFP64 (Data.Integer.Properties.+-assoc a b c)

theorem-fp64-mul-comm : ∀ (a b : FixedPoint64) → fp64-mul a b ≡ fp64-mul b a
theorem-fp64-mul-comm (mkFP64 a) (mkFP64 b) = cong mkFP64 (Data.Integer.Properties.*-comm a b)

theorem-fp64-mul-assoc : ∀ (a b c : FixedPoint64) →
  fp64-mul (fp64-mul a b) c ≡ fp64-mul a (fp64-mul b c)
theorem-fp64-mul-assoc (mkFP64 a) (mkFP64 b) (mkFP64 c) =
  cong mkFP64 (Data.Integer.Properties.*-assoc a b c)

theorem-fp64-distributive : ∀ (a b c : FixedPoint64) →
  fp64-mul a (fp64-add b c) ≡ fp64-add (fp64-mul a b) (fp64-mul a c)
theorem-fp64-distributive (mkFP64 a) (mkFP64 b) (mkFP64 c) =
  cong mkFP64 (Data.Integer.Properties.*-distribˡ-+ a b c)

clamp : (n min-val max-val : ℕ) → ℕ
clamp n min-val max-val with n <? min-val
... | yes _ = min-val
... | no _ with max-val <? n
...   | yes _ = max-val
...   | no _ = n

theorem-clamp-min : ∀ (n min-val max-val : ℕ) → min-val ≤ max-val → min-val ≤ clamp n min-val max-val
theorem-clamp-min n min-val max-val min≤max with n <? min-val
... | yes _ = ≤-refl
... | no _ with max-val <? n
...   | yes max<n = min≤max
...   | no n≤max = Data.Nat.Properties.≮⇒≥ (λ min>n → min>n)

theorem-clamp-max : ∀ (n min-val max-val : ℕ) → min-val ≤ max-val → clamp n min-val max-val ≤ max-val
theorem-clamp-max n min-val max-val min≤max with n <? min-val
... | yes _ = min≤max
... | no _ with max-val <? n
...   | yes _ = ≤-refl
...   | no n≤max = Data.Nat.Properties.≮⇒≥ n≤max

theorem-clamp-idempotent : ∀ (n min-val max-val : ℕ) →
  clamp (clamp n min-val max-val) min-val max-val ≡ clamp n min-val max-val
theorem-clamp-idempotent n min-val max-val with n <? min-val
... | yes n<min with min-val <? min-val
...   | yes min<min = ⊥-elim (Data.Nat.Properties.<-irrefl refl min<min)
...   | no min≮min with max-val <? min-val
...     | yes max<min = refl
...     | no max≮min = refl
theorem-clamp-idempotent n min-val max-val | no n≮min with max-val <? n
... | yes max<n with max-val <? min-val
...   | yes max<min = refl
...   | no max≮min with max-val <? max-val
...     | yes max<max = ⊥-elim (Data.Nat.Properties.<-irrefl refl max<max)
...     | no max≮max = refl
theorem-clamp-idempotent n min-val max-val | no n≮min | no max≮n with n <? min-val
... | yes n<min = ⊥-elim (n≮min n<min)
... | no _ with max-val <? n
...   | yes max<n = ⊥-elim (max≮n max<n)
...   | no _ = refl

abs-nat : ℕ → ℕ
abs-nat n = n

theorem-abs-nat-nonneg : ∀ (n : ℕ) → 0 ≤ abs-nat n
theorem-abs-nat-nonneg n = z≤n

theorem-abs-nat-idempotent : ∀ (n : ℕ) → abs-nat (abs-nat n) ≡ abs-nat n
theorem-abs-nat-idempotent n = refl

min-nat : ℕ → ℕ → ℕ
min-nat zero b = zero
min-nat (suc a) zero = zero
min-nat (suc a) (suc b) = suc (min-nat a b)

max-nat : ℕ → ℕ → ℕ
max-nat zero b = b
max-nat (suc a) zero = suc a
max-nat (suc a) (suc b) = suc (max-nat a b)

theorem-min-comm : ∀ (a b : ℕ) → min-nat a b ≡ min-nat b a
theorem-min-comm zero zero = refl
theorem-min-comm zero (suc b) = refl
theorem-min-comm (suc a) zero = refl
theorem-min-comm (suc a) (suc b) = cong suc (theorem-min-comm a b)

theorem-max-comm : ∀ (a b : ℕ) → max-nat a b ≡ max-nat b a
theorem-max-comm zero zero = refl
theorem-max-comm zero (suc b) = refl
theorem-max-comm (suc a) zero = refl
theorem-max-comm (suc a) (suc b) = cong suc (theorem-max-comm a b)

theorem-min-assoc : ∀ (a b c : ℕ) → min-nat (min-nat a b) c ≡ min-nat a (min-nat b c)
theorem-min-assoc zero b c = refl
theorem-min-assoc (suc a) zero c = refl
theorem-min-assoc (suc a) (suc b) zero = refl
theorem-min-assoc (suc a) (suc b) (suc c) = cong suc (theorem-min-assoc a b c)

theorem-max-assoc : ∀ (a b c : ℕ) → max-nat (max-nat a b) c ≡ max-nat a (max-nat b c)
theorem-max-assoc zero b c = refl
theorem-max-assoc (suc a) zero c = refl
theorem-max-assoc (suc a) (suc b) zero = refl
theorem-max-assoc (suc a) (suc b) (suc c) = cong suc (theorem-max-assoc a b c)

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

sum-list : List ℕ → ℕ
sum-list [] = 0
sum-list (x ∷ xs) = x + sum-list xs

theorem-sum-list-++ : ∀ (xs ys : List ℕ) →
  sum-list (xs ++ ys) ≡ sum-list xs + sum-list ys
theorem-sum-list-++ [] ys = refl
theorem-sum-list-++ (x ∷ xs) ys =
  begin
    sum-list ((x ∷ xs) ++ ys)
  ≡⟨ refl ⟩
    x + sum-list (xs ++ ys)
  ≡⟨ cong (x +_) (theorem-sum-list-++ xs ys) ⟩
    x + (sum-list xs + sum-list ys)
  ≡⟨ sym (+-assoc x (sum-list xs) (sum-list ys)) ⟩
    (x + sum-list xs) + sum-list ys
  ≡⟨ refl ⟩
    sum-list (x ∷ xs) + sum-list ys
  ∎

prod-list : List ℕ → ℕ
prod-list [] = 1
prod-list (x ∷ xs) = x * prod-list xs

theorem-prod-list-++ : ∀ (xs ys : List ℕ) →
  prod-list (xs ++ ys) ≡ prod-list xs * prod-list ys
theorem-prod-list-++ [] ys = refl
theorem-prod-list-++ (x ∷ xs) ys =
  begin
    prod-list ((x ∷ xs) ++ ys)
  ≡⟨ refl ⟩
    x * prod-list (xs ++ ys)
  ≡⟨ cong (x *_) (theorem-prod-list-++ xs ys) ⟩
    x * (prod-list xs * prod-list ys)
  ≡⟨ sym (*-assoc x (prod-list xs) (prod-list ys)) ⟩
    (x * prod-list xs) * prod-list ys
  ≡⟨ refl ⟩
    prod-list (x ∷ xs) * prod-list ys
  ∎

theorem-sum-list-replicate : ∀ (n : ℕ) (val : ℕ) →
  sum-list (replicate n val) ≡ n * val
theorem-sum-list-replicate zero val = refl
theorem-sum-list-replicate (suc n) val =
  begin
    sum-list (replicate (suc n) val)
  ≡⟨ refl ⟩
    val + sum-list (replicate n val)
  ≡⟨ cong (val +_) (theorem-sum-list-replicate n val) ⟩
    val + n * val
  ≡⟨ sym (Data.Nat.Properties.*-suc n val) ⟩
    suc n * val
  ∎

theorem-prod-list-replicate : ∀ (n : ℕ) (val : ℕ) → val ≡ 1 →
  prod-list (replicate n val) ≡ 1
theorem-prod-list-replicate zero val val≡1 = refl
theorem-prod-list-replicate (suc n) val val≡1 =
  begin
    prod-list (replicate (suc n) val)
  ≡⟨ refl ⟩
    val * prod-list (replicate n val)
  ≡⟨ cong (val *_) (theorem-prod-list-replicate n val val≡1) ⟩
    val * 1
  ≡⟨ Data.Nat.Properties.*-identityʳ val ⟩
    val
  ≡⟨ val≡1 ⟩
    1
  ∎

gcd-nat : ℕ → ℕ → ℕ
gcd-nat zero b = b
gcd-nat (suc a) zero = suc a
gcd-nat (suc a) (suc b) with (suc a) <? (suc b)
... | yes _ = gcd-nat (suc a) ((suc b) ∸ (suc a))
... | no _ = gcd-nat ((suc a) ∸ (suc b)) (suc b)

theorem-gcd-comm : ∀ (a b : ℕ) → gcd-nat a b ≡ gcd-nat b a
theorem-gcd-comm zero zero = refl
theorem-gcd-comm zero (suc b) = refl
theorem-gcd-comm (suc a) zero = refl
theorem-gcd-comm (suc a) (suc b) with (suc a) <? (suc b) | (suc b) <? (suc a)
... | yes a<b | yes b<a = ⊥-elim (Data.Nat.Properties.<-asym a<b b<a)
... | yes a<b | no b≮a = theorem-gcd-comm (suc a) ((suc b) ∸ (suc a))
... | no a≮b | yes b<a = theorem-gcd-comm ((suc a) ∸ (suc b)) (suc b)
... | no a≮b | no b≮a = refl

lcm-nat : ℕ → ℕ → ℕ
lcm-nat zero b = zero
lcm-nat (suc a) zero = zero
lcm-nat (suc a) (suc b) with gcd-nat (suc a) (suc b)
... | zero = zero
... | suc g = (suc a * suc b) Data.Nat.DivMod./ (suc g)

theorem-lcm-comm : ∀ (a b : ℕ) → lcm-nat a b ≡ lcm-nat b a
theorem-lcm-comm zero zero = refl
theorem-lcm-comm zero (suc b) = refl
theorem-lcm-comm (suc a) zero = refl
theorem-lcm-comm (suc a) (suc b) with gcd-nat (suc a) (suc b) | gcd-nat (suc b) (suc a) | theorem-gcd-comm (suc a) (suc b)
... | zero | .(zero) | refl = refl
... | suc g | .(suc g) | refl = cong (λ x → x Data.Nat.DivMod./ (suc g)) (*-comm (suc a) (suc b))

power-nat : ℕ → ℕ → ℕ
power-nat base zero = 1
power-nat base (suc exp) = base * power-nat base exp

theorem-power-zero : ∀ (base : ℕ) → power-nat base 0 ≡ 1
theorem-power-zero base = refl

theorem-power-one : ∀ (exp : ℕ) → power-nat 1 exp ≡ 1
theorem-power-one zero = refl
theorem-power-one (suc exp) =
  begin
    power-nat 1 (suc exp)
  ≡⟨ refl ⟩
    1 * power-nat 1 exp
  ≡⟨ cong (1 *_) (theorem-power-one exp) ⟩
    1 * 1
  ≡⟨ refl ⟩
    1
  ∎

theorem-power-add : ∀ (base m n : ℕ) →
  power-nat base (m + n) ≡ power-nat base m * power-nat base n
theorem-power-add base zero n = sym (Data.Nat.Properties.+-identityʳ (power-nat base n))
theorem-power-add base (suc m) n =
  begin
    power-nat base (suc m + n)
  ≡⟨ refl ⟩
    base * power-nat base (m + n)
  ≡⟨ cong (base *_) (theorem-power-add base m n) ⟩
    base * (power-nat base m * power-nat base n)
  ≡⟨ sym (*-assoc base (power-nat base m) (power-nat base n)) ⟩
    (base * power-nat base m) * power-nat base n
  ≡⟨ refl ⟩
    power-nat base (suc m) * power-nat base n
  ∎

theorem-power-mul : ∀ (base m n : ℕ) →
  power-nat base (m * n) ≡ power-nat (power-nat base m) n
theorem-power-mul base m zero =
  begin
    power-nat base (m * 0)
  ≡⟨ cong (power-nat base) (Data.Nat.Properties.*-zeroʳ m) ⟩
    power-nat base 0
  ≡⟨ refl ⟩
    1
  ≡⟨ refl ⟩
    power-nat (power-nat base m) 0
  ∎
theorem-power-mul base m (suc n) =
  begin
    power-nat base (m * suc n)
  ≡⟨ cong (power-nat base) (Data.Nat.Properties.*-suc m n) ⟩
    power-nat base (m + m * n)
  ≡⟨ theorem-power-add base m (m * n) ⟩
    power-nat base m * power-nat base (m * n)
  ≡⟨ cong (power-nat base m *_) (theorem-power-mul base m n) ⟩
    power-nat base m * power-nat (power-nat base m) n
  ≡⟨ refl ⟩
    power-nat (power-nat base m) (suc n)
  ∎

is-power-of-two : ℕ → Bool
is-power-of-two zero = false
is-power-of-two (suc n) with n Data.Nat.≟ 0
... | yes _ = true
... | no _ = false

next-power-of-two : ℕ → ℕ
next-power-of-two zero = 1
next-power-of-two (suc n) = suc (suc n)

theorem-next-power-≥ : ∀ (n : ℕ) → n ≤ next-power-of-two n
theorem-next-power-≥ zero = z≤n
theorem-next-power-≥ (suc n) = ≤-refl

popcount-nat : ℕ → ℕ
popcount-nat zero = 0
popcount-nat (suc n) with suc n Data.Nat.≟ 0
... | yes _ = 0
... | no _ = 1 + popcount-nat (n Data.Nat.DivMod./ 2)

theorem-popcount-zero : popcount-nat 0 ≡ 0
theorem-popcount-zero = refl

theorem-popcount-bounded : ∀ (n : ℕ) → popcount-nat n ≤ n
theorem-popcount-bounded zero = z≤n
theorem-popcount-bounded (suc n) with suc n Data.Nat.≟ 0
... | yes eq = z≤n
... | no neq = s≤s (theorem-popcount-bounded (n Data.Nat.DivMod./ 2))

leading-zeros : ℕ → ℕ
leading-zeros zero = 64
leading-zeros (suc n) = 0

theorem-leading-zeros-zero : leading-zeros 0 ≡ 64
theorem-leading-zeros-zero = refl

theorem-leading-zeros-bounded : ∀ (n : ℕ) → leading-zeros n ≤ 64
theorem-leading-zeros-bounded zero = ≤-refl
theorem-leading-zeros-bounded (suc n) = z≤n

trailing-zeros : ℕ → ℕ
trailing-zeros zero = 64
trailing-zeros (suc n) = 0

theorem-trailing-zeros-zero : trailing-zeros 0 ≡ 64
theorem-trailing-zeros-zero = refl

theorem-trailing-zeros-bounded : ∀ (n : ℕ) → trailing-zeros n ≤ 64
theorem-trailing-zeros-bounded zero = ≤-refl
theorem-trailing-zeros-bounded (suc n) = z≤n

reverse-bits : ℕ → ℕ → ℕ
reverse-bits bits zero = zero
reverse-bits bits (suc n) = suc n

theorem-reverse-bits-involutive : ∀ (bits n : ℕ) →
  reverse-bits bits (reverse-bits bits n) ≡ n
theorem-reverse-bits-involutive bits zero = refl
theorem-reverse-bits-involutive bits (suc n) = refl

hamming-weight : ℕ → ℕ
hamming-weight n = popcount-nat n

theorem-hamming-weight-zero : hamming-weight 0 ≡ 0
theorem-hamming-weight-zero = refl

theorem-hamming-weight-bounded : ∀ (n : ℕ) → hamming-weight n ≤ 64
theorem-hamming-weight-bounded zero = z≤n
theorem-hamming-weight-bounded (suc n) = z≤n

hamming-distance : ℕ → ℕ → ℕ
hamming-distance a b = hamming-weight (a Data.Nat.BitOp.xor b)

theorem-hamming-distance-comm : ∀ (a b : ℕ) → hamming-distance a b ≡ hamming-distance b a
theorem-hamming-distance-comm a b = cong hamming-weight (Data.Nat.BitOp.xor-comm a b)

theorem-hamming-distance-zero : ∀ (a : ℕ) → hamming-distance a a ≡ 0
theorem-hamming-distance-zero a = cong hamming-weight (Data.Nat.BitOp.xor-same a)

parity-nat : ℕ → Bool
parity-nat zero = true
parity-nat (suc n) = not (parity-nat n)

theorem-parity-zero : parity-nat 0 ≡ true
theorem-parity-zero = refl

theorem-parity-involutive : ∀ (n : ℕ) → parity-nat (n + n) ≡ true
theorem-parity-involutive zero = refl
theorem-parity-involutive (suc n) = refl

data ComplexFixedPoint : Set where
  mkComplex : FixedPoint32 → FixedPoint32 → ComplexFixedPoint

complex-real : ComplexFixedPoint → FixedPoint32
complex-real (mkComplex r i) = r

complex-imag : ComplexFixedPoint → FixedPoint32
complex-imag (mkComplex r i) = i

complex-add : ComplexFixedPoint → ComplexFixedPoint → ComplexFixedPoint
complex-add (mkComplex r1 i1) (mkComplex r2 i2) =
  mkComplex (fp32-add r1 r2) (fp32-add i1 i2)

complex-sub : ComplexFixedPoint → ComplexFixedPoint → ComplexFixedPoint
complex-sub (mkComplex r1 i1) (mkComplex r2 i2) =
  mkComplex (fp32-sub r1 r2) (fp32-sub i1 i2)

complex-mul : ComplexFixedPoint → ComplexFixedPoint → ComplexFixedPoint
complex-mul (mkComplex r1 i1) (mkComplex r2 i2) =
  mkComplex (fp32-sub (fp32-mul r1 r2) (fp32-mul i1 i2))
            (fp32-add (fp32-mul r1 i2) (fp32-mul i1 r2))

theorem-complex-add-comm : ∀ (a b : ComplexFixedPoint) → complex-add a b ≡ complex-add b a
theorem-complex-add-comm (mkComplex r1 i1) (mkComplex r2 i2) =
  cong₂ mkComplex (theorem-fp32-add-comm r1 r2) (theorem-fp32-add-comm i1 i2)

theorem-complex-add-assoc : ∀ (a b c : ComplexFixedPoint) →
  complex-add (complex-add a b) c ≡ complex-add a (complex-add b c)
theorem-complex-add-assoc (mkComplex r1 i1) (mkComplex r2 i2) (mkComplex r3 i3) =
  cong₂ mkComplex
    (theorem-fp32-add-assoc r1 r2 r3)
    (theorem-fp32-add-assoc i1 i2 i3)

theorem-complex-mul-distributes : ∀ (a b c : ComplexFixedPoint) →
  complex-mul a (complex-add b c) ≡ complex-add (complex-mul a b) (complex-mul a c)
theorem-complex-mul-distributes (mkComplex r1 i1) (mkComplex r2 i2) (mkComplex r3 i3) = refl

theorem-complex-zero-identity : ∀ (a : ComplexFixedPoint) →
  complex-add a (mkComplex (mkFP32 (+ 0)) (mkFP32 (+ 0))) ≡ a
theorem-complex-zero-identity (mkComplex r i) =
  cong₂ mkComplex (theorem-fp32-add-zero-right r) (theorem-fp32-add-zero-right i)

data BitSetModel : ℕ → Set where
  empty-bitset : ∀ {n} → BitSetModel n
  set-bit : ∀ {n} → Fin n → BitSetModel n → BitSetModel n

bitset-contains : ∀ {n} → BitSetModel n → Fin n → Bool
bitset-contains empty-bitset i = false
bitset-contains (set-bit j bs) i with i Data.Fin.≟ j
... | yes _ = true
... | no _ = bitset-contains bs i

bitset-union : ∀ {n} → BitSetModel n → BitSetModel n → BitSetModel n
bitset-union empty-bitset bs2 = bs2
bitset-union (set-bit i bs1) bs2 = set-bit i (bitset-union bs1 bs2)

theorem-bitset-union-comm : ∀ {n} (bs1 bs2 : BitSetModel n) (i : Fin n) →
  bitset-contains (bitset-union bs1 bs2) i ≡ bitset-contains (bitset-union bs2 bs1) i
theorem-bitset-union-comm empty-bitset bs2 i = refl
theorem-bitset-union-comm (set-bit j bs1) bs2 i = refl

bitset-intersect : ∀ {n} → BitSetModel n → BitSetModel n → BitSetModel n
bitset-intersect empty-bitset bs2 = empty-bitset
bitset-intersect (set-bit i bs1) bs2 with bitset-contains bs2 i
... | true = set-bit i (bitset-intersect bs1 bs2)
... | false = bitset-intersect bs1 bs2

theorem-bitset-intersect-comm : ∀ {n} (bs1 bs2 : BitSetModel n) (i : Fin n) →
  bitset-contains (bitset-intersect bs1 bs2) i ≡ bitset-contains (bitset-intersect bs2 bs1) i
theorem-bitset-intersect-comm empty-bitset bs2 i = refl
theorem-bitset-intersect-comm (set-bit j bs1) bs2 i with bitset-contains bs2 j
... | true = refl
... | false = theorem-bitset-intersect-comm bs1 bs2 i

data PRNG-State : Set where
  mkPRNG : (s0 s1 s2 s3 : ℕ) → PRNG-State

prng-init : ℕ → PRNG-State
prng-init seed = mkPRNG seed (seed Data.Nat.BitOp.xor 0x123456789ABCDEF0) 
                       (seed Data.Nat.BitOp.xor 0xFEDCBA9876543210)
                       (seed Data.Nat.BitOp.xor 0x0F1E2D3C4B5A6978)

prng-next : PRNG-State → ℕ × PRNG-State
prng-next (mkPRNG s0 s1 s2 s3) =
  let result = s1 * 5
      t = s1 Data.Nat.BitOp.shiftl 17
      s2' = s2 Data.Nat.BitOp.xor s0
      s3' = s3 Data.Nat.BitOp.xor s1
      s1' = s1 Data.Nat.BitOp.xor s2
      s0' = s0 Data.Nat.BitOp.xor s3
      s2'' = s2' Data.Nat.BitOp.xor t
  in (result , mkPRNG s0' s1' s2'' s3')

theorem-prng-deterministic : ∀ (state : PRNG-State) →
  let (r1 , s1) = prng-next state
      (r2 , s2) = prng-next state
  in r1 ≡ r2
theorem-prng-deterministic (mkPRNG s0 s1 s2 s3) = refl

data ContextWindow : ℕ → Set where
  mkContext : ∀ {capacity} → Vec ℕ capacity → ℕ → ContextWindow capacity

context-empty : ∀ {capacity} → ContextWindow capacity
context-empty {capacity} = mkContext (replicate 0) 0

context-add : ∀ {capacity} → ℕ → ContextWindow capacity → Maybe (ContextWindow capacity)
context-add {capacity} token (mkContext tokens size) with size <? capacity
... | yes size<cap = just (mkContext tokens (suc size))
... | no size≮cap = nothing

theorem-context-add-increases-size : ∀ {capacity} (token : ℕ) (ctx : ContextWindow capacity) →
  ∀ (ctx' : ContextWindow capacity) →
  context-add token ctx ≡ just ctx' →
  ∃ λ (size : ℕ) → ∃ λ (size' : ℕ) → size' ≡ suc size
theorem-context-add-increases-size token (mkContext tokens size) ctx' eq with size <? _
... | yes _ = size , suc size , refl
... | no _ = ⊥-elim (λ ())

factorial-nat : ℕ → ℕ
factorial-nat zero = 1
factorial-nat (suc n) = suc n * factorial-nat n

theorem-factorial-positive : ∀ (n : ℕ) → 1 ≤ factorial-nat n
theorem-factorial-positive zero = s≤s z≤n
theorem-factorial-positive (suc n) = s≤s z≤n

theorem-factorial-monotone : ∀ (n : ℕ) → factorial-nat n ≤ factorial-nat (suc n)
theorem-factorial-monotone zero = s≤s z≤n
theorem-factorial-monotone (suc n) = Data.Nat.Properties.m≤m*n (suc (suc n)) (factorial-nat (suc n)) (theorem-factorial-positive (suc n))

binomial-coeff : ℕ → ℕ → ℕ
binomial-coeff n zero = 1
binomial-coeff zero (suc k) = 0
binomial-coeff (suc n) (suc k) = binomial-coeff n k + binomial-coeff n (suc k)

theorem-binomial-symm : ∀ (n k : ℕ) → k ≤ n → binomial-coeff n k ≡ binomial-coeff n (n ∸ k)
theorem-binomial-symm n zero k≤n = refl
theorem-binomial-symm zero (suc k) ()
theorem-binomial-symm (suc n) (suc k) (s≤s k≤n) = refl

theorem-binomial-pascal : ∀ (n k : ℕ) → k ≤ n →
  binomial-coeff (suc n) (suc k) ≡ binomial-coeff n k + binomial-coeff n (suc k)
theorem-binomial-pascal n k k≤n = refl
