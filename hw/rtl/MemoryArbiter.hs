{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module MemoryArbiter where

import Clash.Prelude

type Addr = Unsigned 32
type Data = Unsigned 64
type ClientID = Unsigned 4

data MemRequest = MemRequest
    { reqAddr :: Addr
    , reqWrite :: Bool
    , reqData :: Data
    , reqClient :: ClientID
    } deriving (Generic, NFDataX, Show, Eq)

data MemResponse = MemResponse
    { respData :: Data
    , respClient :: ClientID
    , respValid :: Bool
    } deriving (Generic, NFDataX, Show, Eq)

data ArbiterState
    = ArbIdle
    | ArbServing ClientID (Unsigned 8)
    deriving (Generic, NFDataX, Show, Eq)

memoryArbiter
    :: HiddenClockResetEnable dom
    => Vec 4 (Signal dom (Maybe MemRequest))
    -> Signal dom (Maybe MemResponse)
    -> (Signal dom (Maybe MemRequest), Vec 4 (Signal dom (Maybe MemResponse)))
memoryArbiter clientReqs memResp = (memReqOut, clientResps)
  where
    (memReqOut, grantVec) = unbundle $ mealy arbiterT (ArbIdle, 0) (bundle clientReqs)
    clientResps = map (\i -> fmap (filterResp i) memResp) (iterateI (+1) 0)

filterResp :: ClientID -> MemResponse -> Maybe MemResponse
filterResp cid resp
    | respClient resp == cid = Just resp
    | otherwise = Nothing

arbiterT
    :: (ArbiterState, Unsigned 8)
    -> Vec 4 (Maybe MemRequest)
    -> ((ArbiterState, Unsigned 8), (Maybe MemRequest, Vec 4 Bool))
arbiterT (ArbIdle, counter) reqs = case findIndex isJust reqs of
    Just idx -> ((ArbServing (resize (pack idx)) 0, counter + 1), (reqs !! idx, grant))
      where grant = map (\i -> i == idx) (iterateI (+1) 0)
    Nothing -> ((ArbIdle, counter), (Nothing, repeat False))

arbiterT (ArbServing client cycles, counter) reqs
    | cycles < 4 = ((ArbServing client (cycles + 1), counter), (Nothing, repeat False))
    | otherwise = ((ArbIdle, counter), (Nothing, repeat False))

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Vec 4 (Signal System (Maybe MemRequest))
    -> Signal System (Maybe MemResponse)
    -> (Signal System (Maybe MemRequest), Vec 4 (Signal System (Maybe MemResponse)))
topEntity = exposeClockResetEnable memoryArbiter

testInput :: Vec 4 (Signal System (Maybe MemRequest))
testInput = 
    ( pure (Just (MemRequest 0x1000 False 0 0))
    :> pure (Just (MemRequest 0x2000 True 0xDEADBEEF 1))
    :> pure Nothing
    :> pure Nothing
    :> Nil
    )

expectedOutput :: Signal System (Maybe MemRequest) -> Signal System Bool
expectedOutput = outputVerifier' systemClockGen systemResetGen
    ( Just (MemRequest 0x1000 False 0 0)
    :> Just (MemRequest 0x2000 True 0xDEADBEEF 1)
    :> Nothing
    :> Nil
    )

main :: IO ()
main = do
    let outputs = sampleN 10 (fst $ memoryArbiter testInput (pure Nothing))
    print outputs