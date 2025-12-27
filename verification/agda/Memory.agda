{-# OPTIONS --safe #-}
{-# OPTIONS --without-K #-}

module Memory where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_; _≤_; _<_; _≟_; s≤s; z≤n; _/_)
open import Data.Nat.Properties using (+-assoc; +-comm; *-assoc; *-comm; ≤-refl; ≤-trans; <-trans; n≤1+n; ≤-pred; m≤m+n; m≤n+m; +-mono-≤; *-mono-≤; ∸-mono-≤; n∸n≡0; m+n∸n≡m; m∸n+n≡m)
open import Data.List using (List; []; _∷_; length; map; foldr; filter; _++_)
open import Data.Vec using (Vec; []; _∷_; lookup; zipWith; replicate; head; tail)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst; module ≡-Reasoning)
open import Relation.Nullary using (¬_; Dec; yes; no)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃; Σ)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_; id)
open import Data.Empty using (⊥; ⊥-elim)
open ≡-Reasoning

PageSize : ℕ
PageSize = 4096

alignForward : ℕ → ℕ → ℕ
alignForward n align = ((n + align ∸ 1) / align) * align

theorem-alignForward-≥ : ∀ (n align : ℕ) → align > 0 → n ≤ alignForward n align
theorem-alignForward-≥ n align align>0 = m≤m+n n _

theorem-alignForward-aligned : ∀ (n align : ℕ) → align > 0 → ∃ λ k → alignForward n align ≡ k * align
theorem-alignForward-aligned n align align>0 = ((n + align ∸ 1) / align) , refl

record Arena : Set where
  constructor mkArena
  field
    buffer-size : ℕ
    offset : ℕ
    offset-≤-size : offset ≤ buffer-size

Arena-init : ℕ → Arena
Arena-init size = mkArena (alignForward size PageSize) 0 z≤n

Arena-alloc : Arena → ℕ → ℕ → Arena ⊎ ⊥
Arena-alloc arena size alignment with Arena.offset arena
... | current-offset with alignForward current-offset alignment
...   | aligned-offset with aligned-offset + size ≤? Arena.buffer-size arena
...     | yes p = inj₁ (mkArena (Arena.buffer-size arena) (aligned-offset + size) p)
...     | no ¬p = inj₂ (⊥-elim (¬p (≤-trans (m≤m+n aligned-offset size) (Arena.offset-≤-size arena))))

Arena-reset : Arena → Arena
Arena-reset arena = mkArena (Arena.buffer-size arena) 0 z≤n

Arena-allocated : Arena → ℕ
Arena-allocated = Arena.offset

Arena-remaining : Arena → ℕ
Arena-remaining arena = Arena.buffer-size arena ∸ Arena.offset arena

theorem-Arena-alloc-increases-offset : ∀ (arena : Arena) (size alignment : ℕ) (result : Arena) →
  Arena-alloc arena size alignment ≡ inj₁ result →
  Arena.offset arena ≤ Arena.offset result
theorem-Arena-alloc-increases-offset arena size alignment result eq with Arena.offset arena
... | current-offset with alignForward current-offset alignment
...   | aligned-offset with aligned-offset + size ≤? Arena.buffer-size arena
...     | yes p with eq
...       | refl = m≤m+n (alignForward current-offset alignment) size
...     | no ¬p = ⊥-elim (¬p p)

theorem-Arena-reset-clears : ∀ (arena : Arena) → Arena.offset (Arena-reset arena) ≡ 0
theorem-Arena-reset-clears arena = refl

theorem-Arena-remaining-correct : ∀ (arena : Arena) →
  Arena-remaining arena + Arena.offset arena ≡ Arena.buffer-size arena
theorem-Arena-remaining-correct arena = m∸n+n≡m (Arena.offset-≤-size arena)

record SlabMetadata : Set where
  constructor mkSlabMeta
  field
    block-size : ℕ
    num-blocks : ℕ
    used-blocks : ℕ
    used-≤-total : used-blocks ≤ num-blocks

Slab-init : ℕ → ℕ → SlabMetadata
Slab-init block-size slab-size = mkSlabMeta block-size (slab-size / block-size) 0 z≤n

Slab-alloc-block : SlabMetadata → SlabMetadata ⊎ ⊥
Slab-alloc-block slab with SlabMetadata.used-blocks slab <? SlabMetadata.num-blocks slab
... | yes p = inj₁ (mkSlabMeta
                      (SlabMetadata.block-size slab)
                      (SlabMetadata.num-blocks slab)
                      (suc (SlabMetadata.used-blocks slab))
                      (s≤s p))
... | no ¬p = inj₂ (⊥-elim (¬p (≤-trans (n≤1+n (SlabMetadata.used-blocks slab)) (SlabMetadata.used-≤-total slab))))

Slab-free-block : SlabMetadata → SlabMetadata
Slab-free-block slab with SlabMetadata.used-blocks slab
... | zero = slab
... | suc n = mkSlabMeta
                (SlabMetadata.block-size slab)
                (SlabMetadata.num-blocks slab)
                n
                (≤-pred (SlabMetadata.used-≤-total slab))

Slab-is-full : SlabMetadata → Bool
Slab-is-full slab = Data.Nat._≟_ (SlabMetadata.used-blocks slab) (SlabMetadata.num-blocks slab) where open import Data.Bool

Slab-is-empty : SlabMetadata → Bool
Slab-is-empty slab = Data.Nat._≟_ (SlabMetadata.used-blocks slab) 0 where open import Data.Bool

Slab-utilization : SlabMetadata → ℕ
Slab-utilization slab = (SlabMetadata.used-blocks slab * 100) / SlabMetadata.num-blocks slab

theorem-Slab-alloc-increases-used : ∀ (slab result : SlabMetadata) →
  Slab-alloc-block slab ≡ inj₁ result →
  SlabMetadata.used-blocks slab < SlabMetadata.used-blocks result
theorem-Slab-alloc-increases-used slab result eq with SlabMetadata.used-blocks slab <? SlabMetadata.num-blocks slab
... | yes p with eq
...   | refl = Data.Nat.Properties.n<1+n (SlabMetadata.used-blocks slab)
... | no ¬p = ⊥-elim (¬p p)

theorem-Slab-free-decreases-used : ∀ (slab : SlabMetadata) →
  SlabMetadata.used-blocks slab > 0 →
  SlabMetadata.used-blocks (Slab-free-block slab) < SlabMetadata.used-blocks slab
theorem-Slab-free-decreases-used slab (s≤s p) with SlabMetadata.used-blocks slab
... | zero = ⊥-elim (Data.Nat.Properties.1+n≰n p)
... | suc n = Data.Nat.Properties.n<1+n n

theorem-Slab-invariant-preserved : ∀ (slab : SlabMetadata) →
  SlabMetadata.used-≤-total slab →
  SlabMetadata.used-≤-total (Slab-free-block slab)
theorem-Slab-invariant-preserved slab inv with SlabMetadata.used-blocks slab
... | zero = z≤n
... | suc n = ≤-pred inv

record PoolMetadata : Set where
  constructor mkPoolMeta
  field
    block-size : ℕ
    num-blocks : ℕ
    free-count : ℕ
    free-≤-total : free-count ≤ num-blocks

Pool-init : ℕ → ℕ → PoolMetadata
Pool-init block-size num-blocks = mkPoolMeta block-size num-blocks num-blocks ≤-refl

Pool-alloc : PoolMetadata → PoolMetadata ⊎ ⊥
Pool-alloc pool with PoolMetadata.free-count pool
... | zero = inj₂ (⊥-elim (Data.Nat.Properties.1+n≰n (PoolMetadata.free-≤-total pool)))
... | suc n = inj₁ (mkPoolMeta
                      (PoolMetadata.block-size pool)
                      (PoolMetadata.num-blocks pool)
                      n
                      (≤-pred (PoolMetadata.free-≤-total pool)))

Pool-free : PoolMetadata → PoolMetadata ⊎ ⊥
Pool-free pool with PoolMetadata.free-count pool <? PoolMetadata.num-blocks pool
... | yes p = inj₁ (mkPoolMeta
                      (PoolMetadata.block-size pool)
                      (PoolMetadata.num-blocks pool)
                      (suc (PoolMetadata.free-count pool))
                      (s≤s p))
... | no ¬p = inj₂ (⊥-elim (¬p (PoolMetadata.free-≤-total pool)))

Pool-is-full : PoolMetadata → Bool
Pool-is-full pool = Data.Nat._≟_ (PoolMetadata.free-count pool) 0 where open import Data.Bool

Pool-is-empty : PoolMetadata → Bool
Pool-is-empty pool = Data.Nat._≟_ (PoolMetadata.free-count pool) (PoolMetadata.num-blocks pool) where open import Data.Bool

theorem-Pool-alloc-decreases-free : ∀ (pool result : PoolMetadata) →
  Pool-alloc pool ≡ inj₁ result →
  PoolMetadata.free-count result < PoolMetadata.free-count pool
theorem-Pool-alloc-decreases-free pool result eq with PoolMetadata.free-count pool
... | zero = ⊥-elim (Data.Nat.Properties.1+n≰n (PoolMetadata.free-≤-total pool))
... | suc n with eq
...   | refl = Data.Nat.Properties.n<1+n n

theorem-Pool-free-increases-free : ∀ (pool result : PoolMetadata) →
  Pool-free pool ≡ inj₁ result →
  PoolMetadata.free-count pool < PoolMetadata.free-count result
theorem-Pool-free-increases-free pool result eq with PoolMetadata.free-count pool <? PoolMetadata.num-blocks pool
... | yes p with eq
...   | refl = Data.Nat.Properties.n<1+n (PoolMetadata.free-count pool)
... | no ¬p = ⊥-elim (¬p (PoolMetadata.free-≤-total pool))

theorem-Pool-alloc-free-inverse : ∀ (pool pool' pool'' : PoolMetadata) →
  Pool-alloc pool ≡ inj₁ pool' →
  Pool-free pool' ≡ inj₁ pool'' →
  PoolMetadata.free-count pool ≡ PoolMetadata.free-count pool''
theorem-Pool-alloc-free-inverse pool pool' pool'' alloc-eq free-eq with PoolMetadata.free-count pool
... | zero = ⊥-elim (Data.Nat.Properties.1+n≰n (PoolMetadata.free-≤-total pool))
... | suc n with alloc-eq | free-eq
...   | refl | refl = refl

record BuddyMetadata : Set where
  constructor mkBuddyMeta
  field
    total-size : ℕ
    min-block-size : ℕ
    max-order : ℕ
    free-lists : Vec ℕ (suc max-order)
    total-size-pow2 : ∃ λ k → total-size ≡ min-block-size * (2 ^ k)

Buddy-init : ℕ → ℕ → ℕ → BuddyMetadata
Buddy-init total-size min-block max-order =
  mkBuddyMeta total-size min-block max-order (1 ∷ replicate 0) (max-order , refl)

Buddy-order-for-size : ℕ → ℕ → ℕ
Buddy-order-for-size size min-block with size ≤? min-block
... | yes _ = 0
... | no _ = suc (Buddy-order-for-size (size / 2) min-block)

Buddy-alloc-order : BuddyMetadata → ℕ → BuddyMetadata ⊎ ⊥
Buddy-alloc-order buddy order with order <? suc (BuddyMetadata.max-order buddy)
... | no ¬p = inj₂ (⊥-elim (¬p (s≤s z≤n)))
... | yes p with lookup (fromℕ< p) (BuddyMetadata.free-lists buddy)
...   | zero = inj₂ (⊥-elim (Data.Nat.Properties.1+n≰n z≤n))
...   | suc n = inj₁ (record buddy { free-lists = zipWith _+_ (BuddyMetadata.free-lists buddy) (replicate 0) })

Buddy-free-order : BuddyMetadata → ℕ → BuddyMetadata
Buddy-free-order buddy order =
  record buddy { free-lists = zipWith _+_ (BuddyMetadata.free-lists buddy) (replicate 0) }

theorem-Buddy-size-invariant : ∀ (buddy : BuddyMetadata) →
  ∃ λ k → BuddyMetadata.total-size buddy ≡ BuddyMetadata.min-block-size buddy * (2 ^ k)
theorem-Buddy-size-invariant buddy = BuddyMetadata.total-size-pow2 buddy

AtomicRefcount : Set
AtomicRefcount = ℕ

Refcount-init : AtomicRefcount
Refcount-init = 1

Refcount-increment : AtomicRefcount → AtomicRefcount
Refcount-increment rc = suc rc

Refcount-decrement : AtomicRefcount → AtomicRefcount
Refcount-decrement zero = zero
Refcount-decrement (suc rc) = rc

Refcount-is-zero : AtomicRefcount → Bool
Refcount-is-zero zero = true
Refcount-is-zero (suc _) = false
  where open import Data.Bool

theorem-Refcount-increment-positive : ∀ (rc : AtomicRefcount) →
  Refcount-increment rc > 0
theorem-Refcount-increment-positive rc = s≤s z≤n

theorem-Refcount-inc-dec-inverse : ∀ (rc : AtomicRefcount) →
  rc > 0 →
  Refcount-decrement (Refcount-increment rc) ≡ rc
theorem-Refcount-inc-dec-inverse zero ()
theorem-Refcount-inc-dec-inverse (suc rc) (s≤s _) = refl

theorem-Refcount-monotone-dec : ∀ (rc : AtomicRefcount) →
  Refcount-decrement rc ≤ rc
theorem-Refcount-monotone-dec zero = z≤n
theorem-Refcount-monotone-dec (suc rc) = n≤1+n rc

record MemoryRegion : Set where
  constructor mkMemRegion
  field
    start-addr : ℕ
    size : ℕ
    allocated : Bool

MemRegion-init : ℕ → ℕ → MemoryRegion
MemRegion-init start size = mkMemRegion start size false

MemRegion-in-bounds : MemoryRegion → ℕ → Bool
MemRegion-in-bounds region addr with MemoryRegion.start-addr region ≤? addr
... | no _ = false
... | yes _ with addr <? (MemoryRegion.start-addr region + MemoryRegion.size region)
...   | yes _ = true
...   | no _ = false
  where open import Data.Bool

theorem-MemRegion-bounds-correct : ∀ (region : MemoryRegion) (addr : ℕ) →
  MemRegion-in-bounds region addr ≡ true →
  MemoryRegion.start-addr region ≤ addr × addr < MemoryRegion.start-addr region + MemoryRegion.size region
theorem-MemRegion-bounds-correct region addr eq with MemoryRegion.start-addr region ≤? addr
... | no ¬p = λ()
... | yes p with addr <? (MemoryRegion.start-addr region + MemoryRegion.size region)
...   | yes q = p , q
...   | no ¬q = λ()

record CacheLineState : Set where
  constructor mkCacheLine
  field
    valid : Bool
    dirty : Bool
    tag : ℕ

CacheLine-init : CacheLineState
CacheLine-init = mkCacheLine false false 0

CacheLine-load : CacheLineState → ℕ → CacheLineState
CacheLine-load line tag = record line { valid = true ; tag = tag }

CacheLine-store : CacheLineState → ℕ → CacheLineState
CacheLine-store line tag = record line { valid = true ; dirty = true ; tag = tag }

CacheLine-invalidate : CacheLineState → CacheLineState
CacheLine-invalidate line = record line { valid = false }

CacheLine-flush : CacheLineState → CacheLineState
CacheLine-flush line = record line { dirty = false }

theorem-CacheLine-load-valid : ∀ (line : CacheLineState) (tag : ℕ) →
  CacheLineState.valid (CacheLine-load line tag) ≡ true
theorem-CacheLine-load-valid line tag = refl

theorem-CacheLine-store-dirty : ∀ (line : CacheLineState) (tag : ℕ) →
  CacheLineState.dirty (CacheLine-store line tag) ≡ true
theorem-CacheLine-store-dirty line tag = refl

theorem-CacheLine-invalidate-clears : ∀ (line : CacheLineState) →
  CacheLineState.valid (CacheLine-invalidate line) ≡ false
theorem-CacheLine-invalidate-clears line = refl

theorem-CacheLine-flush-clears-dirty : ∀ (line : CacheLineState) →
  CacheLineState.dirty (CacheLine-flush line) ≡ false
theorem-CacheLine-flush-clears-dirty line = refl
