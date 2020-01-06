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

public @RefAware
class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
  @Nonnull
  private final TensorList[] input;
  @Nonnull
  private final Layer layer;
  private boolean calcDerivatives = true;
  private TensorList[] inputDerivative;
  private TensorList output;
  private DeltaSet<UUID> layerDerivative;

  public SimpleListEval(@Nonnull final Layer layer, @Nonnull final TensorList... input) {
    {
      Layer temp_09_0001 = layer == null ? null : layer.addRef();
      this.layer = temp_09_0001 == null ? null : temp_09_0001.addRef();
      if (null != temp_09_0001)
        temp_09_0001.freeRef();
    }
    layer.freeRef();
    {
      TensorList[] temp_09_0002 = TensorList
          .addRefs(input);
      this.input = TensorList.addRefs(temp_09_0002);
      if (null != temp_09_0002)
        ReferenceCounting.freeRefs(temp_09_0002);
    }
    ReferenceCounting.freeRefs(input);
    {
      DeltaSet<UUID> temp_09_0003 = new DeltaSet<UUID>();
      if (null != layerDerivative)
        layerDerivative.freeRef();
      layerDerivative = temp_09_0003 == null ? null : temp_09_0003.addRef();
      if (null != temp_09_0003)
        temp_09_0003.freeRef();
    }
  }

  @Override
  public TensorList[] getInputDerivative() {
    return TensorList.addRefs(inputDerivative);
  }

  public DeltaSet<UUID> getLayerDerivative() {
    return layerDerivative == null ? null : layerDerivative.addRef();
  }

  public SimpleListEval setLayerDerivative(DeltaSet<UUID> layerDerivative) {
    {
      DeltaSet<UUID> temp_09_0004 = layerDerivative == null ? null
          : layerDerivative.addRef();
      if (null != this.layerDerivative)
        this.layerDerivative.freeRef();
      this.layerDerivative = temp_09_0004 == null ? null : temp_09_0004.addRef();
      if (null != temp_09_0004)
        temp_09_0004.freeRef();
    }
    if (null != layerDerivative)
      layerDerivative.freeRef();
    return this.addRef();
  }

  @Override
  public TensorList getOutput() {
    return output == null ? null : output.addRef();
  }

  public boolean isCalcDerivatives() {
    return calcDerivatives;
  }

  public SimpleListEval setCalcDerivatives(boolean calcDerivatives) {
    this.calcDerivatives = calcDerivatives;
    return this.addRef();
  }

  public static void accumulate(@Nonnull final TensorList buffer, @Nonnull final TensorList data) {
    RefIntStream.range(0, data.length())
        .forEach(RefUtil.wrapInterface(b -> {
          @Nullable
          Tensor r = data.get(b);
          @Nullable
          Tensor l = buffer.get(b);
          l.addInPlace(r == null ? null : r.addRef());
          if (null != l)
            l.freeRef();
          if (null != r)
            r.freeRef();
        }, data == null ? null : data, buffer == null ? null : buffer));
  }

  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, final TensorList... tensor) {
    SimpleResult temp_09_0012 = run(layer == null ? null : layer, true,
        TensorList.addRefs(tensor));
    if (null != tensor)
      ReferenceCounting.freeRefs(tensor);
    return temp_09_0012;
  }

  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, boolean calcDerivatives, final TensorList... tensor) {
    SimpleListEval temp_09_0015 = new SimpleListEval(layer == null ? null : layer,
        TensorList.addRefs(tensor));
    SimpleListEval temp_09_0016 = temp_09_0015.setCalcDerivatives(calcDerivatives);
    SimpleResult temp_09_0013 = temp_09_0016.call();
    if (null != temp_09_0016)
      temp_09_0016.freeRef();
    if (null != temp_09_0015)
      temp_09_0015.freeRef();
    if (null != tensor)
      ReferenceCounting.freeRefs(tensor);
    return temp_09_0013;
  }

  public static @SuppressWarnings("unused")
  SimpleListEval[] addRefs(SimpleListEval[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleListEval::addRef)
        .toArray((x) -> new SimpleListEval[x]);
  }

  public static @SuppressWarnings("unused")
  SimpleListEval[][] addRefs(SimpleListEval[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleListEval::addRefs)
        .toArray((x) -> new SimpleListEval[x][]);
  }

  @Nonnull
  @Override
  public SimpleResult call() {
    TensorList[] inputCopy = RefArrays.stream(TensorList.addRefs(input)).map(x -> {
      TensorList temp_09_0008 = x.copy();
      if (null != x)
        x.freeRef();
      return temp_09_0008;
    }).toArray(i -> new TensorList[i]);
    {
      TensorList[] temp_09_0005 = RefArrays
          .stream(TensorList.addRefs(inputCopy)).map(tensorList -> {
            TensorArray temp_09_0009 = new TensorArray(tensorList.stream().map(i -> {
              Tensor temp_09_0010 = new Tensor(i.getDimensions());
              if (null != i)
                i.freeRef();
              return temp_09_0010;
            }).toArray(i -> new Tensor[i]));
            if (null != tensorList)
              tensorList.freeRef();
            return temp_09_0009;
          }).toArray(i -> new TensorList[i]);
      if (null != inputDerivative)
        ReferenceCounting.freeRefs(inputDerivative);
      inputDerivative = TensorList.addRefs(temp_09_0005);
      if (null != temp_09_0005)
        ReferenceCounting.freeRefs(temp_09_0005);
    }
    Result[] inputs = RefIntStream.range(0, inputCopy.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Result>) i -> {
          return new Result(inputCopy[i], new Result.Accumulator() {
            @Override
            public void accept(DeltaSet<UUID> buffer, TensorList data) {
              SimpleListEval.accumulate(inputDerivative[i], data);
            }
          }) {
            @Override
            public boolean isAlive() {
              return true;
            }

            public @SuppressWarnings("unused")
            void _free() {
            }
          };
        }, TensorList.addRefs(inputCopy))).toArray(i -> new Result[i]);
    if (null != inputCopy)
      ReferenceCounting.freeRefs(inputCopy);
    @Nullable final Result eval = layer.eval(Result.addRefs(inputs));
    if (null != inputs)
      ReferenceCounting.freeRefs(inputs);
    TensorList temp_09_0017 = eval.getData();
    TensorList outputData = temp_09_0017.copy();
    if (null != temp_09_0017)
      temp_09_0017.freeRef();
    RefUtil.freeRef(eval.getData());
    {
      DeltaSet<UUID> temp_09_0006 = new DeltaSet<>();
      if (null != this.layerDerivative)
        this.layerDerivative.freeRef();
      this.layerDerivative = temp_09_0006 == null ? null : temp_09_0006.addRef();
      if (null != temp_09_0006)
        temp_09_0006.freeRef();
    }
    if (isCalcDerivatives())
      eval.accumulate(layerDerivative == null ? null : layerDerivative.addRef(),
          getFeedback(outputData == null ? null : outputData.addRef()));
    if (null != eval)
      eval.freeRef();
    {
      TensorList temp_09_0007 = outputData == null ? null : outputData.addRef();
      if (null != output)
        output.freeRef();
      output = temp_09_0007 == null ? null : temp_09_0007.addRef();
      if (null != temp_09_0007)
        temp_09_0007.freeRef();
    }
    if (null != outputData)
      outputData.freeRef();
    return this.addRef();
  }

  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    TensorArray temp_09_0014 = new TensorArray(data.stream().map(t -> {
      Tensor temp_09_0011 = t.map(v -> 1.0);
      if (null != t)
        t.freeRef();
      return temp_09_0011;
    }).toArray(i -> new Tensor[i]));
    data.freeRef();
    return temp_09_0014;
  }

  public void _free() {
    if (null != layerDerivative)
      layerDerivative.freeRef();
    layerDerivative = null;
    if (null != output)
      output.freeRef();
    output = null;
    if (null != inputDerivative)
      ReferenceCounting.freeRefs(inputDerivative);
    inputDerivative = null;
    layer.freeRef();
    ReferenceCounting.freeRefs(input);
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleListEval addRef() {
    return (SimpleListEval) super.addRef();
  }
}
