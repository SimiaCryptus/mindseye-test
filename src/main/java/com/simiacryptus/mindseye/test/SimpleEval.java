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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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
    Layer temp_01_0001 = layer == null ? null : layer.addRef();
    this.layer = temp_01_0001 == null ? null : temp_01_0001.addRef();
    if (null != temp_01_0001)
      temp_01_0001.freeRef();
    layer.freeRef();
    Tensor[] temp_01_0002 = Tensor.addRefs(input);
    this.input = Tensor.addRefs(temp_01_0002);
    if (null != temp_01_0002)
      ReferenceCounting.freeRefs(temp_01_0002);
    ReferenceCounting.freeRefs(input);
    Tensor[] temp_01_0003 = null;
    if (null != this.derivative)
      ReferenceCounting.freeRefs(this.derivative);
    this.derivative = Tensor.addRefs(temp_01_0003);
    if (null != temp_01_0003)
      ReferenceCounting.freeRefs(temp_01_0003);
    Tensor temp_01_0004 = null;
    if (null != this.output)
      this.output.freeRef();
    this.output = temp_01_0004 == null ? null : temp_01_0004.addRef();
    if (null != temp_01_0004)
      temp_01_0004.freeRef();
  }

  @Nullable
  public Tensor[] getDerivative() {
    return Tensor.addRefs(derivative);
  }

  @Nullable
  public Tensor getOutput() {
    return output == null ? null : output.addRef();
  }

  public boolean isCalcDerivative() {
    return calcDerivative;
  }

  public SimpleEval setValidateDerivative(boolean calcDerivative) {
    this.calcDerivative = calcDerivative;
    return this.addRef();
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, final Tensor... tensor) {
    SimpleEval temp_01_0012 = run(layer == null ? null : layer, true, Tensor.addRefs(tensor));
    if (null != tensor)
      ReferenceCounting.freeRefs(tensor);
    return temp_01_0012;
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, boolean validateDerivative, final Tensor... tensor) {
    SimpleEval temp_01_0015 = new SimpleEval(layer == null ? null : layer, Tensor.addRefs(tensor));
    SimpleEval temp_01_0016 = temp_01_0015.setValidateDerivative(validateDerivative);
    SimpleEval temp_01_0013 = temp_01_0016.call();
    if (null != temp_01_0016)
      temp_01_0016.freeRef();
    if (null != temp_01_0015)
      temp_01_0015.freeRef();
    if (null != tensor)
      ReferenceCounting.freeRefs(tensor);
    return temp_01_0013;
  }

  public static @SuppressWarnings("unused") SimpleEval[] addRefs(SimpleEval[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleEval::addRef).toArray((x) -> new SimpleEval[x]);
  }

  public static @SuppressWarnings("unused") SimpleEval[][] addRefs(SimpleEval[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleEval::addRefs).toArray((x) -> new SimpleEval[x][]);
  }

  @Nonnull
  @Override
  public SimpleEval call() {
    Tensor[] inputCopy = RefArrays.stream(Tensor.addRefs(input)).map(x -> {
      Tensor temp_01_0009 = x.copy();
      if (null != x)
        x.freeRef();
      return temp_01_0009;
    }).toArray(i -> new Tensor[i]);
    Tensor[] temp_01_0005 = RefArrays.stream(Tensor.addRefs(inputCopy)).map(input -> {
      Tensor temp_01_0010 = new Tensor(input.getDimensions());
      if (null != input)
        input.freeRef();
      return temp_01_0010;
    }).toArray(i -> new Tensor[i]);
    if (null != derivative)
      ReferenceCounting.freeRefs(derivative);
    derivative = Tensor.addRefs(temp_01_0005);
    if (null != temp_01_0005)
      ReferenceCounting.freeRefs(temp_01_0005);
    Result[] input = RefIntStream.range(0, inputCopy.length).mapToObj(RefUtil.wrapInterface((IntFunction<Result>) i -> {
      return new Result(new TensorArray(inputCopy[i].addRef()), new Result.Accumulator() {
        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList data) {
          buffer.freeRef();
          data.stream().forEach(t -> {
            derivative[i].addInPlace(t);
          });
          data.freeRef();
        }
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }

        public void _free() {

        }
      };
    }, Tensor.addRefs(inputCopy))).toArray(i -> new Result[i]);
    if (null != inputCopy)
      ReferenceCounting.freeRefs(inputCopy);
    @Nullable
    final Result eval;
    try {
      eval = layer.eval(Result.addRefs(input));
    } finally {
      for (@Nonnull
      Result result : input) {
        RefUtil.freeRef(result.getData());
      }
    }
    if (null != input)
      ReferenceCounting.freeRefs(input);
    TensorList evalData = eval.getData();
    TensorList outputTensorList = evalData.copy();
    if (null != evalData)
      evalData.freeRef();
    @Nullable
    Tensor outputTensor = outputTensorList.get(0);
    @Nonnull
    DeltaSet<UUID> deltaSet = new DeltaSet<>();
    synchronized (this) {
      if (null != output) {
        Tensor temp_01_0006 = null;
        if (null != output)
          output.freeRef();
        output = temp_01_0006 == null ? null : temp_01_0006.addRef();
        if (null != temp_01_0006)
          temp_01_0006.freeRef();
      }
    }
    Tensor temp_01_0007 = outputTensor.copy();
    if (null != output)
      output.freeRef();
    output = temp_01_0007 == null ? null : temp_01_0007.addRef();
    if (null != temp_01_0007)
      temp_01_0007.freeRef();
    if (isCalcDerivative()) {
      eval.accumulate(deltaSet == null ? null : deltaSet.addRef(),
          getFeedback(outputTensorList == null ? null : outputTensorList.addRef()));
    }
    if (null != eval)
      eval.freeRef();
    if (null != outputTensorList)
      outputTensorList.freeRef();
    if (null != outputTensor)
      outputTensor.freeRef();
    deltaSet.freeRef();
    return this.addRef();
  }

  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    TensorArray temp_01_0014 = new TensorArray(data.stream().map(t -> {
      Tensor temp_01_0011 = t.map(v -> 1.0);
      if (null != t)
        t.freeRef();
      return temp_01_0011;
    }).toArray(i -> new Tensor[i]));
    data.freeRef();
    return temp_01_0014;
  }

  public void _free() {
    if (null != derivative)
      ReferenceCounting.freeRefs(derivative);
    derivative = null;
    layer.freeRef();
    ReferenceCounting.freeRefs(input);
    synchronized (this) {
      if (null != output) {
        output.freeRef();
        output = null;
      }
    }
  }

  public @Override @SuppressWarnings("unused") SimpleEval addRef() {
    return (SimpleEval) super.addRef();
  }
}
