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
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

public class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
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
    IntStream.range(0, data.length()).forEach(b -> {
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

  @Nonnull
  @Override
  public SimpleResult call() {
    TensorList[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new TensorList[i]);
    inputDerivative = Arrays.stream(inputCopy).map(tensorList -> new TensorArray(tensorList.stream().map(i -> {
              return new Tensor(i.getDimensions());
            }).toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
    Result[] inputs = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new Result(inputCopy[i], (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
        SimpleListEval.accumulate(inputDerivative[i], data);
      }) {
        @Override
        public boolean isAlive() {
          return true;
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

  @Override
  protected void _free() {
  }
}
