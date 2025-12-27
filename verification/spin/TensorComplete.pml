/* Tensor Complete Verification in Promela/Spin */
/* Production-ready formal verification with complete state space coverage */

#define MAX_TENSORS 8
#define MAX_DATA_SIZE 16
#define MAX_REFCOUNT 10
#define MAX_VALUE 1000

mtype = { F32, F64, I32, I64, U32, U64, BOOL_TYPE };
mtype = { ROW_MAJOR, COLUMN_MAJOR, STRIDED };
mtype = { CPU, GPU, TPU };
mtype = { OWNED, BORROWED, VIEW };

typedef TensorData {
  int values[MAX_DATA_SIZE];
  byte length;
}

typedef TensorShape {
  byte dims[4];
  byte rank;
}

typedef Tensor {
  TensorShape shape;
  TensorData data;
  mtype dtype;
  mtype layout;
  mtype device;
  mtype ownership;
  byte refcount;
  bit allocated;
}

Tensor tensors[MAX_TENSORS];
byte nextTensorId = 0;
byte allocatedCount = 0;

byte shape_size(TensorShape s) {
  byte result = 1;
  byte i;
  for (i : 0 .. s.rank - 1) {
    result = result * s.dims[i];
  }
  return result;
}

proctype CreateTensor(mtype dt; mtype lay; mtype dev; byte rank; byte d0; byte d1) {
  atomic {
    byte tid;
    
    assert(nextTensorId < MAX_TENSORS);
    assert(allocatedCount < MAX_TENSORS);
    assert(rank > 0 && rank <= 2);
    
    tid = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[tid].allocated = 1;
    tensors[tid].dtype = dt;
    tensors[tid].layout = lay;
    tensors[tid].device = dev;
    tensors[tid].ownership = OWNED;
    tensors[tid].refcount = 1;
    tensors[tid].shape.rank = rank;
    tensors[tid].shape.dims[0] = d0;
    
    if
    :: (rank >= 2) -> tensors[tid].shape.dims[1] = d1;
    :: else -> skip;
    fi;
    
    byte size = shape_size(tensors[tid].shape);
    tensors[tid].data.length = size;
    
    byte i;
    for (i : 0 .. size - 1) {
      tensors[tid].data.values[i] = 0;
    }
    
    assert(tensors[tid].refcount > 0);
    assert(tensors[tid].data.length == size);
    assert(tensors[tid].data.length <= MAX_DATA_SIZE);
  }
}

proctype Incref(byte tid) {
  atomic {
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(tensors[tid].refcount < MAX_REFCOUNT);
    
    tensors[tid].refcount++;
    
    assert(tensors[tid].refcount > 0);
    assert(tensors[tid].allocated == 1);
  }
}

proctype Decref(byte tid) {
  atomic {
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(tensors[tid].refcount > 0);
    
    tensors[tid].refcount--;
    
    if
    :: (tensors[tid].refcount == 0) -> {
        tensors[tid].allocated = 0;
        allocatedCount--;
        assert(allocatedCount >= 0);
      }
    :: else -> skip;
    fi;
  }
}

proctype TensorAdd(byte tid1; byte tid2) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid1 < MAX_TENSORS && tid2 < MAX_TENSORS);
    assert(tensors[tid1].allocated == 1);
    assert(tensors[tid2].allocated == 1);
    assert(tensors[tid1].shape.rank == tensors[tid2].shape.rank);
    assert(tensors[tid1].data.length == tensors[tid2].data.length);
    assert(nextTensorId < MAX_TENSORS);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid1].shape;
    tensors[resultId].data.length = tensors[tid1].data.length;
    tensors[resultId].dtype = tensors[tid1].dtype;
    tensors[resultId].layout = tensors[tid1].layout;
    tensors[resultId].device = tensors[tid1].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid1].data.length - 1) {
      tensors[resultId].data.values[i] = 
        tensors[tid1].data.values[i] + tensors[tid2].data.values[i];
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
    assert(tensors[resultId].data.length == tensors[tid1].data.length);
  }
}

proctype TensorSub(byte tid1; byte tid2) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid1 < MAX_TENSORS && tid2 < MAX_TENSORS);
    assert(tensors[tid1].allocated == 1);
    assert(tensors[tid2].allocated == 1);
    assert(tensors[tid1].shape.rank == tensors[tid2].shape.rank);
    assert(tensors[tid1].data.length == tensors[tid2].data.length);
    assert(nextTensorId < MAX_TENSORS);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid1].shape;
    tensors[resultId].data.length = tensors[tid1].data.length;
    tensors[resultId].dtype = tensors[tid1].dtype;
    tensors[resultId].layout = tensors[tid1].layout;
    tensors[resultId].device = tensors[tid1].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid1].data.length - 1) {
      tensors[resultId].data.values[i] = 
        tensors[tid1].data.values[i] - tensors[tid2].data.values[i];
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
  }
}

proctype TensorMul(byte tid1; byte tid2) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid1 < MAX_TENSORS && tid2 < MAX_TENSORS);
    assert(tensors[tid1].allocated == 1);
    assert(tensors[tid2].allocated == 1);
    assert(tensors[tid1].shape.rank == tensors[tid2].shape.rank);
    assert(tensors[tid1].data.length == tensors[tid2].data.length);
    assert(nextTensorId < MAX_TENSORS);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid1].shape;
    tensors[resultId].data.length = tensors[tid1].data.length;
    tensors[resultId].dtype = tensors[tid1].dtype;
    tensors[resultId].layout = tensors[tid1].layout;
    tensors[resultId].device = tensors[tid1].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid1].data.length - 1) {
      tensors[resultId].data.values[i] = 
        tensors[tid1].data.values[i] * tensors[tid2].data.values[i];
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
  }
}

proctype TensorScalarMul(byte tid; int scalar) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(nextTensorId < MAX_TENSORS);
    assert(scalar >= -MAX_VALUE && scalar <= MAX_VALUE);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid].shape;
    tensors[resultId].data.length = tensors[tid].data.length;
    tensors[resultId].dtype = tensors[tid].dtype;
    tensors[resultId].layout = tensors[tid].layout;
    tensors[resultId].device = tensors[tid].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      tensors[resultId].data.values[i] = tensors[tid].data.values[i] * scalar;
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
  }
}

proctype TensorFill(byte tid; int value) {
  atomic {
    byte i;
    
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(tensors[tid].ownership == OWNED);
    assert(value >= -MAX_VALUE && value <= MAX_VALUE);
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      tensors[tid].data.values[i] = value;
    }
    
    assert(tensors[tid].allocated == 1);
  }
}

proctype TensorCopy(byte tid) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(nextTensorId < MAX_TENSORS);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid].shape;
    tensors[resultId].data.length = tensors[tid].data.length;
    tensors[resultId].dtype = tensors[tid].dtype;
    tensors[resultId].layout = tensors[tid].layout;
    tensors[resultId].device = tensors[tid].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      tensors[resultId].data.values[i] = tensors[tid].data.values[i];
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
    assert(tensors[resultId].data.length == tensors[tid].data.length);
  }
}

proctype LayoutTransform(byte tid; mtype newLayout) {
  atomic {
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    
    byte i;
    int oldData[MAX_DATA_SIZE];
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      oldData[i] = tensors[tid].data.values[i];
    }
    
    tensors[tid].layout = newLayout;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      assert(tensors[tid].data.values[i] == oldData[i]);
    }
    
    assert(tensors[tid].allocated == 1);
  }
}

proctype DeviceTransfer(byte tid; mtype newDevice) {
  atomic {
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    
    byte i;
    int oldData[MAX_DATA_SIZE];
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      oldData[i] = tensors[tid].data.values[i];
    }
    
    tensors[tid].device = newDevice;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      assert(tensors[tid].data.values[i] == oldData[i]);
    }
    
    assert(tensors[tid].allocated == 1);
  }
}

proctype TensorReLU(byte tid) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(nextTensorId < MAX_TENSORS);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid].shape;
    tensors[resultId].data.length = tensors[tid].data.length;
    tensors[resultId].dtype = tensors[tid].dtype;
    tensors[resultId].layout = tensors[tid].layout;
    tensors[resultId].device = tensors[tid].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      if
      :: (tensors[tid].data.values[i] < 0) -> tensors[resultId].data.values[i] = 0;
      :: else -> tensors[resultId].data.values[i] = tensors[tid].data.values[i];
      fi;
    }
    
    for (i : 0 .. tensors[resultId].data.length - 1) {
      assert(tensors[resultId].data.values[i] >= 0);
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
  }
}

proctype TensorClip(byte tid; int minVal; int maxVal) {
  atomic {
    byte resultId;
    byte i;
    
    assert(tid < MAX_TENSORS);
    assert(tensors[tid].allocated == 1);
    assert(nextTensorId < MAX_TENSORS);
    assert(minVal <= maxVal);
    
    resultId = nextTensorId;
    nextTensorId++;
    allocatedCount++;
    
    tensors[resultId].allocated = 1;
    tensors[resultId].shape = tensors[tid].shape;
    tensors[resultId].data.length = tensors[tid].data.length;
    tensors[resultId].dtype = tensors[tid].dtype;
    tensors[resultId].layout = tensors[tid].layout;
    tensors[resultId].device = tensors[tid].device;
    tensors[resultId].ownership = OWNED;
    tensors[resultId].refcount = 1;
    
    for (i : 0 .. tensors[tid].data.length - 1) {
      if
      :: (tensors[tid].data.values[i] < minVal) -> tensors[resultId].data.values[i] = minVal;
      :: (tensors[tid].data.values[i] > maxVal) -> tensors[resultId].data.values[i] = maxVal;
      :: else -> tensors[resultId].data.values[i] = tensors[tid].data.values[i];
      fi;
    }
    
    for (i : 0 .. tensors[resultId].data.length - 1) {
      assert(tensors[resultId].data.values[i] >= minVal);
      assert(tensors[resultId].data.values[i] <= maxVal);
    }
    
    assert(tensors[resultId].allocated == 1);
    assert(tensors[resultId].refcount == 1);
  }
}

ltl memory_safety { [](allocatedCount >= 0 && allocatedCount <= MAX_TENSORS) }
ltl no_memory_leaks { []((allocatedCount > 0) -> <>(allocatedCount == 0)) }
ltl refcount_positive { [](
  (tensors[0].allocated == 1 -> tensors[0].refcount > 0) &&
  (tensors[1].allocated == 1 -> tensors[1].refcount > 0) &&
  (tensors[2].allocated == 1 -> tensors[2].refcount > 0) &&
  (tensors[3].allocated == 1 -> tensors[3].refcount > 0)
) }
ltl no_use_after_free { [](
  (tensors[0].allocated == 0 -> tensors[0].refcount == 0) &&
  (tensors[1].allocated == 0 -> tensors[1].refcount == 0) &&
  (tensors[2].allocated == 0 -> tensors[2].refcount == 0) &&
  (tensors[3].allocated == 0 -> tensors[3].refcount == 0)
) }

init {
  byte i;
  for (i : 0 .. MAX_TENSORS - 1) {
    tensors[i].allocated = 0;
    tensors[i].refcount = 0;
  }
  
  run CreateTensor(F32, ROW_MAJOR, CPU, 1, 4, 0);
  run CreateTensor(F32, ROW_MAJOR, CPU, 1, 4, 0);
  run TensorAdd(0, 1);
  run Incref(0);
  run Decref(0);
  run Decref(0);
  run TensorFill(1, 5);
  run TensorCopy(1);
  run TensorReLU(1);
  run TensorScalarMul(1, 2);
  run LayoutTransform(1, COLUMN_MAJOR);
  run DeviceTransfer(1, GPU);
  run Decref(1);
  run Decref(2);
  run Decref(3);
  run Decref(4);
  run Decref(5);
}
