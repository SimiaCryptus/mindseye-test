/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.function.IntFunction;

public class SimpleEval extends ReferenceCountingBase implements Callable<SimpleEval> {
  @Nonnull
  private final Tensor[] input;
  @Nonnull
  private final Layer layer;
  private boolean calcDerivative = false;
  @Nullable
  private Tensor[] derivative;
  @Nullable
  private Tensor output;

  public SimpleEval(@Nonnull final Layer layer, @Nonnull final Tensor... input) {
    this.layer = layer;
    this.input = input;
    this.output = null;
  }

  @Nullable
  public Tensor[] getDerivative() {
    return RefUtil.addRefs(derivative);
  }

  @Nullable
  public Tensor getOutput() {
    return output == null ? null : output.addRef();
  }

  public boolean isCalcDerivative() {
    return calcDerivative;
  }

  public void setValidateDerivative(boolean calcDerivative) {
    this.calcDerivative = calcDerivative;
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, @Nullable final Tensor... tensor) {
    return run(layer, true, tensor);
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, boolean validateDerivative, @Nullable final Tensor... tensor) {
    SimpleEval simpleEval = new SimpleEval(layer, tensor);
    try {
      simpleEval.setValidateDerivative(validateDerivative);
      return simpleEval.call();
    } finally {
      simpleEval.freeRef();
    }
  }


  @Nonnull
  @Override
  public SimpleEval call() {
    eval();
    return this.addRef();
  }

  public void eval() {
    setResult(eval(input()));
  }

  public void setResult(Result eval) {
    assert eval != null;
    TensorList evalData = eval.getData();
    Tensor outputTensor = evalData.get(0);
    synchronized (this) {
      if (null != output) {
        output.freeRef();
      }
      output = outputTensor.copy();
    }
    outputTensor.freeRef();
    if (isCalcDerivative()) {
      eval.accumulate(new DeltaSet<>(), getFeedback(evalData));
    } else {
      evalData.freeRef();
    }
    eval.freeRef();
  }

  @NotNull
  public Result[] input() {
    Tensor[] inputCopy = RefArrays.stream(RefUtil.addRefs(input)).map(x -> {
      try {
        return x.copy();
      } finally {
        x.freeRef();
      }
    }).toArray(Tensor[]::new);
    if (null != derivative)
      RefUtil.freeRef(derivative);
    derivative = RefArrays.stream(RefUtil.addRefs(inputCopy)).map(input1 -> {
      try {
        return input1.getDimensions();
      } finally {
        input1.freeRef();
      }
    }).map(Tensor::new).toArray(Tensor[]::new);
    return RefIntStream.range(0, inputCopy.length).mapToObj(RefUtil.wrapInterface((IntFunction<Result>) i -> {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          RefUtil.addRefs(derivative);
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          buffer.freeRef();
          data.stream().forEach(t -> {
            derivative[i].addInPlace(t);
          });
          data.freeRef();
        }

        @Override
        public void _free() {
          RefUtil.freeRef(derivative);
          super._free();
        }
      };
      return new Result(new TensorArray(inputCopy[i].addRef()), accumulator) {
        @Override
        public boolean isAlive() {
          return true;
        }

        @Override
        public void _free() {
          super._free();
        }
      };
    }, inputCopy)).toArray(Result[]::new);
  }

  public Result eval(Result[] input) {
    return layer.eval(input);
  }

  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    TensorArray temp_01_0014 = new TensorArray(data.stream().map(t -> {
      Tensor temp_01_0011 = t.map(v -> 1.0);
      t.freeRef();
      return temp_01_0011;
    }).toArray(Tensor[]::new));
    data.freeRef();
    return temp_01_0014;
  }

  public void _free() {
    super._free();
    if (null != derivative)
      RefUtil.freeRef(derivative);
    derivative = null;
    layer.freeRef();
    RefUtil.freeRef(input);
    synchronized (this) {
      if (null != output) {
        output.freeRef();
        output = null;
      }
    }
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SimpleEval addRef() {
    return (SimpleEval) super.addRef();
  }
}
