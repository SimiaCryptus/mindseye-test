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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.Callable;

public @RefAware
class SimpleListEval extends ReferenceCountingBase
    implements Callable<SimpleResult>, SimpleResult {
  @Nonnull
  private final TensorList[] input;
  @Nonnull
  private final Layer layer;
  private boolean calcDerivatives = true;
  private TensorList[] inputDerivative;
  private TensorList output;
  private DeltaSet<UUID> layerDerivative;

  public SimpleListEval(@Nonnull final Layer layer, @Nonnull final TensorList... input) {
    this.layer = layer;
    this.input = input;
    layerDerivative = new DeltaSet<UUID>();
  }

  @Override
  public TensorList[] getInputDerivative() {
    return inputDerivative;
  }

  public DeltaSet<UUID> getLayerDerivative() {
    return layerDerivative;
  }

  public SimpleListEval setLayerDerivative(DeltaSet<UUID> layerDerivative) {
    this.layerDerivative = layerDerivative;
    return this;
  }

  @Override
  public TensorList getOutput() {
    return output;
  }

  public boolean isCalcDerivatives() {
    return calcDerivatives;
  }

  public SimpleListEval setCalcDerivatives(boolean calcDerivatives) {
    this.calcDerivatives = calcDerivatives;
    return this;
  }

  public static void accumulate(@Nonnull final TensorList buffer, @Nonnull final TensorList data) {
    RefIntStream.range(0, data.length()).forEach(b -> {
      @Nullable
      Tensor r = data.get(b);
      @Nullable
      Tensor l = buffer.get(b);
      l.addInPlace(r);
    });
  }

  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, final TensorList... tensor) {
    return run(layer, true, tensor);
  }

  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, boolean calcDerivatives, final TensorList... tensor) {
    return new SimpleListEval(layer, tensor).setCalcDerivatives(calcDerivatives).call();
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
    TensorList[] inputCopy = RefArrays.stream(input).map(x -> x.copy())
        .toArray(i -> new TensorList[i]);
    inputDerivative = RefArrays.stream(inputCopy)
        .map(tensorList -> new TensorArray(tensorList.stream().map(i -> {
          return new Tensor(i.getDimensions());
        }).toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
    Result[] inputs = RefIntStream.range(0, inputCopy.length).mapToObj(i -> {
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
    }).toArray(i -> new Result[i]);
    @Nullable final Result eval = layer.eval(inputs);
    TensorList outputData = eval.getData().copy();
    eval.getData();
    this.layerDerivative = new DeltaSet<>();
    if (isCalcDerivatives())
      eval.accumulate(layerDerivative, getFeedback(outputData));
    output = outputData;
    return this;
  }

  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    return new TensorArray(data.stream().map(t -> {
      return t.map(v -> 1.0);
    }).toArray(i -> new Tensor[i]));
  }

  public void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleListEval addRef() {
    return (SimpleListEval) super.addRef();
  }
}
