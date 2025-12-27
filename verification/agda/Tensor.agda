{-# OPTIONS –without-K #-}

module TensorZig where

open import Agda.Builtin.Float using (Float ; primFloatPlus ; primFloatTimes)
open import Relation.Binary.PropositionalEquality using (≡ ; refl ; sym ; trans ; cong ; subst)
open import Data.Nat using (ℕ ; zero ; suc ; + ; * ; ≤ ; < ; z≤n ; s≤s)
open import Data.Vec using (Vec ; [] ; ∷ ; replicate ; map ; lookup ; updateAt ; tabulate ; concat ; splitAt ; drop)
open import Data.Sum using (⊎ ; inj₁ ; inj₂)
open import Data.Product using (× ; , ; proj₁ ; proj₂)
open import Data.Fin as F using (Fin ; toℕ)
open import Data.Vec.All as VA using (All)

infixl 6 +ᶠ
infixl 7 *ᶠ

+ᶠ : Float → Float → Float
+ᶠ = primFloatPlus

*ᶠ : Float → Float → Float
*ᶠ = primFloatTimes

0ᶠ : Float
0ᶠ = 0.0

1ℕ : ℕ
1ℕ = suc zero

≤-trans : ∀ {a b c : ℕ} → a ≤ b → b ≤ c → a ≤ c
≤-trans z≤n _ = z≤n
≤-trans (s≤s p) (s≤s q) = s≤s (≤-trans p q)

<-≤-trans : ∀ {a b c : ℕ} → a < b → b ≤ c → a < c
<-≤-trans p q = ≤-trans p q

+-mono-< : ∀ {a b c : ℕ} → a < b → c + a < c + b
+-mono-< {c = zero} p = p
+-mono-< {c = suc c} p = s≤s (+-mono-< {c = c} p)

toℕ< : ∀ {n : ℕ} → Fin n → toℕ {n} _ < n
toℕ< {suc n} F.zero = s≤s z≤n
toℕ< {suc n} (F.suc i) = s≤s (toℕ< i)

fromNat< : ∀ {n : ℕ} → (m : ℕ) → m < n → Fin n
fromNat< {zero} m ()
fromNat< {suc n} zero p = F.zero
fromNat< {suc n} (suc m) (s≤s p) = F.suc (fromNat< {n} m p)

data All₂ {A : Set} (R : A → A → Set) : ∀ {n : ℕ} → Vec A n → Vec A n → Set where
[]₂ : All₂ R [] []
∷₂ : ∀ {n : ℕ} {x y : A} {xs ys : Vec A n} → R x y → All₂ R xs ys → All₂ R (x ∷ xs) (y ∷ ys)

prod : ∀ {n : ℕ} → Vec ℕ n → ℕ
prod [] = suc zero
prod (d ∷ ds) = d * prod ds

strides : ∀ {n : ℕ} → Vec ℕ n → Vec ℕ n
strides [] = []
strides (d ∷ ds) = prod ds ∷ strides ds

Arr : ∀ {n : ℕ} → Vec ℕ n → Set
Arr [] = Float
Arr (d ∷ ds) = Vec (Arr ds) d

Ix : ∀ {n : ℕ} → Vec ℕ n → Set
Ix ds = VA.All (λ d → Fin d) ds

record Shape (n : ℕ) : Set where
constructor mkShape
field
dims : Vec ℕ n
pos : VA.All (λ d → 0 < d) dims

record Tensor {n : ℕ} (sh : Shape (suc n)) : Set where
constructor mkTensor
field
arr : Arr (Shape.dims sh)

zeroArr : ∀ {n : ℕ} → (ds : Vec ℕ n) → Arr ds
zeroArr [] = 0ᶠ
zeroArr (d ∷ ds) = replicate d (zeroArr ds)

init : ∀ {n : ℕ} → (sh : Shape (suc n)) → Tensor sh
init sh = mkTensor (zeroArr (Shape.dims sh))

getArr : ∀ {n : ℕ} {ds : Vec ℕ n} → Arr ds → Ix ds → Float
getArr {ds = []} x VA.[] = x
getArr {ds = d ∷ ds} xs (VA.∷ i is) = getArr (lookup i xs) is

setArr : ∀ {n : ℕ} {ds : Vec ℕ n} → Arr ds → Ix ds → Float → Arr ds
setArr {ds = []} x VA.[] v = v
setArr {ds = d ∷ ds} xs (VA.∷ i is) v = updateAt i (λ sub → setArr sub is v) xs

get : ∀ {n : ℕ} {sh : Shape (suc n)} → Tensor sh → Ix (Shape.dims sh) → Float
get t ix = getArr (Tensor.arr t) ix

set : ∀ {n : ℕ} {sh : Shape (suc n)} → Tensor sh → Ix (Shape.dims sh) → Float → Tensor sh
set t ix v = mkTensor (setArr (Tensor.arr t) ix v)

flatten : ∀ {n : ℕ} {ds : Vec ℕ n} → Arr ds → Vec Float (prod ds)
flatten {ds = []} x = x ∷ []
flatten {ds = d ∷ ds} xs = concat (map flatten xs)

unflattenVec : ∀ {n : ℕ} (d : ℕ) (ds : Vec ℕ n) → Vec Float (d * prod ds) → Vec (Arr ds) d
unflattenVec zero ds v = []
unflattenVec (suc d) ds v =
let p = prod ds in
let sp = splitAt p v in
let chunk = proj₁ sp in
let rest = proj₂ sp in
unflatten ds chunk ∷ unflattenVec d ds rest

unflatten : ∀ {n : ℕ} (ds : Vec ℕ n) → Vec Float (prod ds) → Arr ds
unflatten [] (x ∷ []) = x
unflatten (d ∷ ds) v = unflattenVec d ds v

reshape : ∀ {n m : ℕ} {sh₁ : Shape (suc n)} {sh₂ : Shape (suc m)} →
Tensor sh₁ → prod (Shape.dims sh₁) ≡ prod (Shape.dims sh₂) → Tensor sh₂
reshape {sh₁ = sh₁} {sh₂ = sh₂} t eq =
let flat₁ = flatten (Tensor.arr t) in
let flat₂ = subst (Vec Float) eq flat₁ in
mkTensor (unflatten (Shape.dims sh₂) flat₂)

CompatDim : ℕ → ℕ → Set
CompatDim sd td = (sd ≡ 1ℕ) ⊎ (sd ≡ td)

BroadcastCompat : ∀ {r off : ℕ} → Vec ℕ r → Vec ℕ (off + r) → Set
BroadcastCompat {r} {off} s t = All₂ CompatDim s (drop off t)

dropAll : ∀ {A : Set} {P : A → Set} (k : ℕ) {n : ℕ} {xs : Vec A (k + n)} →
VA.All P xs → VA.All P (drop k xs)
dropAll zero all = all
dropAll (suc k) (VA.∷ p ps) = dropAll k ps

broadcastFin : ∀ {sd td : ℕ} → CompatDim sd td → Fin td → Fin sd
broadcastFin (inj₁ eq) _ = subst Fin (sym eq) F.zero
broadcastFin (inj₂ eq) i = subst Fin (sym eq) i

broadcastIx : ∀ {r : ℕ} {s t : Vec ℕ r} → All₂ CompatDim s t → Ix t → Ix s
broadcastIx []₂ VA.[] = VA.[]
broadcastIx (c ∷₂ cs) (VA.∷ i is) = VA.∷ (broadcastFin c i) (broadcastIx cs is)

allFin : (n : ℕ) → Vec (Fin n) n
allFin zero = []
allFin (suc n) = F.zero ∷ map F.suc (allFin n)

allIndices : ∀ {n : ℕ} (ds : Vec ℕ n) → Vec (Ix ds) (prod ds)
allIndices [] = VA.[] ∷ []
allIndices (d ∷ ds) =
concat (map (λ i → map (λ is → VA.∷ i is) (allIndices ds)) (allFin d))

broadcast : ∀ {n off : ℕ} {shS : Shape (suc n)} {shT : Shape (off + suc n)} →
Tensor shS → BroadcastCompat (Shape.dims shS) (Shape.dims shT) → Tensor shT
broadcast {off = off} {shS = shS} {shT = shT} t bc =
let dsT = Shape.dims shT in
let idxT = allIndices dsT in
let flat = map (λ ixT → get t (broadcastIx bc (dropAll off ixT))) idxT in
mkTensor (unflatten dsT flat)

record Range (d : ℕ) : Set where
constructor mkRange
field
start : ℕ
len : ℕ
lenPos : 0 < len
startLen≤ : start + len ≤ d

sliceDims : ∀ {n : ℕ} {ds : Vec ℕ n} → VA.All (λ d → Range d) ds → Vec ℕ n
sliceDims {ds = []} VA.[] = []
sliceDims {ds = d ∷ ds} (VA.∷ r rs) = Range.len r ∷ sliceDims rs

slicePos : ∀ {n : ℕ} {ds : Vec ℕ n} → (rs : VA.All (λ d → Range d) ds) → VA.All (λ d → 0 < d) (sliceDims rs)
slicePos {ds = []} VA.[] = VA.[]
slicePos {ds = d ∷ ds} (VA.∷ r rs) = VA.∷ (Range.lenPos r) (slicePos rs)

sliceShape : ∀ {n : ℕ} {sh : Shape (suc n)} → (rs : VA.All (λ d → Range d) (Shape.dims sh)) → Shape (suc n)
sliceShape {sh = sh} rs = mkShape (sliceDims rs) (slicePos rs)

indexInRange : ∀ {d : ℕ} → Range d → Fin (Range.len _) → Fin d
indexInRange {d = d} r i =
let a = Range.start r in
let l = Range.len r in
let p1 = toℕ< i in
let p2 = +-mono-< {c = a} p1 in
let p3 = <-≤-trans p2 (Range.startLen≤ r) in
fromNat< (a + toℕ i) p3

sliceIx : ∀ {n : ℕ} {ds : Vec ℕ n} →
(rs : VA.All (λ d → Range d) ds) →
Ix (sliceDims rs) → Ix ds
sliceIx {ds = []} VA.[] VA.[] = VA.[]
sliceIx {ds = d ∷ ds} (VA.∷ r rs) (VA.∷ i is) =
VA.∷ (indexInRange r i) (sliceIx rs is)

slice : ∀ {n : ℕ} {sh : Shape (suc n)} →
Tensor sh → (rs : VA.All (λ d → Range d) (Shape.dims sh)) → Tensor (sliceShape rs)
slice {sh = sh} t rs =
let shN = sliceShape {sh = sh} rs in
let dsN = Shape.dims shN in
let idxN = allIndices dsN in
let flat = map (λ ixN → get t (sliceIx rs ixN)) idxN in
mkTensor (unflatten dsN flat)

sumFin : ∀ {k : ℕ} → (Fin k → Float) → Float
sumFin {zero} f = 0ᶠ
sumFin {suc k} f = f F.zero +ᶠ sumFin (λ i → f (F.suc i))

matmulShape : (shA : Shape (suc (suc zero))) → (shB : Shape (suc (suc zero))) → Shape (suc (suc zero))
matmulShape (mkShape (m ∷ k ∷ []) (VA.∷ pm (VA.∷ pk VA.[])))
(mkShape (k’ ∷ n ∷ []) (VA.∷ pk’ (VA.∷ pn VA.[]))) =
mkShape (m ∷ n ∷ []) (VA.∷ pm (VA.∷ pn VA.[]))

matmul : ∀ {shA shB : Shape (suc (suc zero))} →
Tensor shA → Tensor shB →
let dsA = Shape.dims shA in
let dsB = Shape.dims shB in
(proj₁ (proj₂ (dsA , dsB))) ≡ proj₁ dsB →
Tensor (matmulShape shA shB)
matmul {shA = mkShape (m ∷ k ∷ []) (VA.∷ pm (VA.∷ pk VA.[]))}
{shB = mkShape (k’ ∷ n ∷ []) (VA.∷ pk’ (VA.∷ pn VA.[]))}
a b kEq =
let aArr = Tensor.arr a in
let bArr0 = Tensor.arr b in
let bArr = subst (λ x → Vec (Vec Float n) x) (sym kEq) bArr0 in
let cArr =
tabulate (λ i →
tabulate (λ j →
sumFin (λ t →
lookup t (lookup i aArr) *ᶠ lookup j (lookup t bArr)
)
)
)
in
mkTensor cArr