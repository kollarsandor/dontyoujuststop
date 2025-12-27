---- MODULE DistributedTraining ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumGPUs,
    MaxIterations,
    MaxGradientValue

ASSUME NumGPUs \in Nat /\ NumGPUs > 0
ASSUME MaxIterations \in Nat /\ MaxIterations > 0
ASSUME MaxGradientValue \in Nat /\ MaxGradientValue > 0

GPUs == 1..NumGPUs

VARIABLES
    gpuState,
    weights,
    gradients,
    allreduceBuffer,
    syncStep,
    messages,
    acked,
    iteration

vars == <<gpuState, weights, gradients, allreduceBuffer, syncStep, messages, acked, iteration>>

GPUStates == {"idle", "computing", "syncing", "waiting", "done"}

TypeInvariant ==
    /\ gpuState \in [GPUs -> GPUStates]
    /\ weights \in [GPUs -> [1..100 -> Nat]]
    /\ gradients \in [GPUs -> [1..100 -> Nat]]
    /\ allreduceBuffer \in [GPUs -> [1..100 -> Nat]]
    /\ syncStep \in Nat
    /\ messages \in [GPUs -> [GPUs -> Seq(Nat)]]
    /\ acked \in [GPUs -> [GPUs -> BOOLEAN]]
    /\ iteration \in Nat

Init ==
    /\ gpuState = [g \in GPUs |-> "idle"]
    /\ weights = [g \in GPUs |-> [i \in 1..100 |-> 1]]
    /\ gradients = [g \in GPUs |-> [i \in 1..100 |-> 0]]
    /\ allreduceBuffer = [g \in GPUs |-> [i \in 1..100 |-> 0]]
    /\ syncStep = 0
    /\ messages = [g1 \in GPUs |-> [g2 \in GPUs |-> <<>>]]
    /\ acked = [g1 \in GPUs |-> [g2 \in GPUs |-> FALSE]]
    /\ iteration = 0

ComputeGradient(g) ==
    /\ gpuState[g] = "idle"
    /\ iteration < MaxIterations
    /\ gpuState' = [gpuState EXCEPT ![g] = "computing"]
    /\ gradients' = [gradients EXCEPT ![g] = [i \in 1..100 |-> 
        IF weights[g][i] > 0 THEN (weights[g][i] - 1) ELSE 0]]
    /\ UNCHANGED <<weights, allreduceBuffer, syncStep, messages, acked, iteration>>

SendGradient(sender, receiver) ==
    /\ gpuState[sender] = "computing"
    /\ sender # receiver
    /\ messages' = [messages EXCEPT ![sender][receiver] = 
        Append(messages[sender][receiver], syncStep)]
    /\ gpuState' = [gpuState EXCEPT ![sender] = "syncing"]
    /\ UNCHANGED <<weights, gradients, allreduceBuffer, syncStep, acked, iteration>>

ReceiveGradient(receiver, sender) ==
    /\ gpuState[receiver] \in {"syncing", "waiting"}
    /\ Len(messages[sender][receiver]) > 0
    /\ LET msg == Head(messages[sender][receiver])
       IN /\ msg = syncStep
          /\ messages' = [messages EXCEPT ![sender][receiver] = 
              Tail(messages[sender][receiver])]
          /\ acked' = [acked EXCEPT ![receiver][sender] = TRUE]
          /\ allreduceBuffer' = [allreduceBuffer EXCEPT ![receiver] = 
              [i \in 1..100 |-> 
                IF allreduceBuffer[receiver][i] + gradients[sender][i] <= MaxGradientValue
                THEN allreduceBuffer[receiver][i] + gradients[sender][i]
                ELSE MaxGradientValue]]
    /\ UNCHANGED <<gpuState, weights, gradients, syncStep, iteration>>

AllReduce(g) ==
    /\ gpuState[g] = "syncing"
    /\ \A other \in GPUs \ {g} : acked[g][other] = TRUE
    /\ weights' = [weights EXCEPT ![g] = [i \in 1..100 |->
        IF allreduceBuffer[g][i] > 0
        THEN weights[g][i] - (allreduceBuffer[g][i] \div NumGPUs)
        ELSE weights[g][i]]]
    /\ allreduceBuffer' = [allreduceBuffer EXCEPT ![g] = [i \in 1..100 |-> 0]]
    /\ acked' = [acked EXCEPT ![g] = [other \in GPUs |-> FALSE]]
    /\ gpuState' = [gpuState EXCEPT ![g] = "waiting"]
    /\ UNCHANGED <<gradients, syncStep, messages, iteration>>

Synchronize ==
    /\ \A g \in GPUs : gpuState[g] = "waiting"
    /\ syncStep' = syncStep + 1
    /\ iteration' = iteration + 1
    /\ gpuState' = [g \in GPUs |-> 
        IF iteration' >= MaxIterations THEN "done" ELSE "idle"]
    /\ UNCHANGED <<weights, gradients, allreduceBuffer, messages, acked>>

Next ==
    \/ \E g \in GPUs : ComputeGradient(g)
    \/ \E s, r \in GPUs : SendGradient(s, r)
    \/ \E r, s \in GPUs : ReceiveGradient(r, s)
    \/ \E g \in GPUs : AllReduce(g)
    \/ Synchronize

Spec == Init /\ [][Next]_vars

SafetyProperty ==
    \A g \in GPUs : \A i \in 1..100 : weights[g][i] >= 0

LivenessProperty ==
    <>(iteration = MaxIterations)

EventualConsistency ==
    (iteration = MaxIterations) =>
    \A g1, g2 \in GPUs : weights[g1] = weights[g2]

NoDeadlock ==
    \/ \E g \in GPUs : gpuState[g] # "done"
    \/ iteration = MaxIterations

MessageOrdering ==
    \A sender, receiver \in GPUs :
        \A i, j \in 1..Len(messages[sender][receiver]) :
            i < j => messages[sender][receiver][i] <= messages[sender][receiver][j]

THEOREM SpecImpliesTypeInvariant == Spec => []TypeInvariant

THEOREM SpecImpliesSafety == Spec => []SafetyProperty

THEOREM FairSpecImpliesLiveness ==
    Spec /\ WF_vars(Next) => LivenessProperty

THEOREM FairSpecImpliesConsistency ==
    Spec /\ WF_vars(Next) => <>EventualConsistency

THEOREM SpecImpliesNoDeadlock == Spec => []NoDeadlock

THEOREM SpecImpliesMessageOrdering == Spec => []MessageOrdering

====
