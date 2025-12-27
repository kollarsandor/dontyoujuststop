{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module RankerCore where

import Clash.Prelude
import Data.Word

type Score = Unsigned 32
type SegmentID = Word64
type Position = Word64
type QueryHash = Word64

data RankRequest = RankRequest
    { queryHash :: QueryHash
    , segmentID :: SegmentID
    , segmentPos :: Position
    , baseScore :: Score
    } deriving (Generic, NFDataX, Show, Eq)

data RankResult = RankResult
    { resultID :: SegmentID
    , finalScore :: Score
    , rank :: Unsigned 16
    } deriving (Generic, NFDataX, Show, Eq)

rankerCore
    :: HiddenClockResetEnable dom
    => Signal dom (Maybe RankRequest)
    -> Signal dom (Maybe RankResult)
rankerCore = mealy rankerT (0, 0, Nothing)

rankerT
    :: (Unsigned 16, Score, Maybe QueryHash)
    -> Maybe RankRequest
    -> ((Unsigned 16, Score, Maybe QueryHash), Maybe RankResult)
rankerT (counter, _, lastQuery) Nothing = ((counter, 0, lastQuery), Nothing)
rankerT (counter, _, lastQuery) (Just req) = ((newCounter, final, Just (queryHash req)), Just result)
  where
    newCounter = if lastQuery == Just (queryHash req) then counter + 1 else 1
    pos64 :: Word64
    pos64 = segmentPos req
    
    positionBias :: Score
    positionBias = resize $ truncateB ((1000 :: Word64) `div` (pos64 + 1))
    
    final :: Score
    final = baseScore req + positionBias
    
    result :: RankResult
    result = RankResult (segmentID req) final newCounter

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System (Maybe RankRequest)
    -> Signal System (Maybe RankResult)
topEntity = exposeClockResetEnable rankerCore

testRankRequest :: RankRequest
testRankRequest = RankRequest
    { queryHash = 0x123456789ABCDEF0
    , segmentID = 0xFEDCBA9876543210
    , segmentPos = 10
    , baseScore = 1000
    }

simulateRanker :: Maybe RankRequest -> (Unsigned 16, Score)
simulateRanker Nothing = (0, 0)
simulateRanker (Just req) = 
    let pos64 = segmentPos req
        positionBias = resize $ truncateB ((1000 :: Word64) `div` (pos64 + 1))
        final = baseScore req + positionBias
    in (1, final)

main :: IO ()
main = do
    let (rankCount, score) = simulateRanker (Just testRankRequest)
    print rankCount
    print score
    let positions = [1, 10, 100, 1000 :: Word64]
    mapM_ (\pos -> do
        let bias = truncateB ((1000 :: Word64) `div` (pos + 1)) :: Score
        print bias
        ) positions
    let segments = 
            [ RankRequest 0x1 0x100 5 800
            , RankRequest 0x1 0x200 15 900
            , RankRequest 0x1 0x300 50 700
            ]
    mapM_ (\req -> do
        let (_, finalScore) = simulateRanker (Just req)
        print finalScore
        ) segments