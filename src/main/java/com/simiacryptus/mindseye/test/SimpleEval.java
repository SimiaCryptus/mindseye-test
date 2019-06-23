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

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

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
    this.derivative = null;
    this.output = null;
    for (@Nonnull Tensor x : input) x.addRef();
    layer.addRef();
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, final Tensor... tensor) {
    return run(layer, true, tensor);
  }

  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, boolean validateDerivative, final Tensor... tensor) {
    return new SimpleEval(layer, tensor).setValidateDerivative(validateDerivative).call();
  }

  @Override
  protected void _free() {
    for (@Nonnull Tensor x : input) x.freeRef();
    layer.freeRef();
    if (null != derivative) for (@Nonnull Tensor x : derivative) x.freeRef();
    synchronized (this) {
      if (null != output) {
        output.freeRef();
        output = null;
      }
    }
  }

  @Nonnull
  @Override
  public SimpleEval call() {
    Tensor[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new Tensor[i]);
    derivative = Arrays.stream(inputCopy).map(input -> new Tensor(input.getDimensions())).toArray(i -> new Tensor[i]);
    Result[] input = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new Result(TensorArray.create(inputCopy[i]), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
        data.stream().forEach(t -> {
          derivative[i].addInPlace(t);
          t.freeRef();
        });
        data.freeRef();
      }) {
        @Override
        protected void _free() {

        }

        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new Result[i]);
    @Nullable final Result eval;
    try {
      eval = layer.eval(input);
    } finally {
      for (@Nonnull Result result : input) {
        result.getData().freeRef();
        result.freeRef();
      }
      for (@Nonnull Tensor tensor : inputCopy) {
        tensor.freeRef();
      }
    }
    TensorList evalData = eval.getData();
    TensorList outputTensorList = evalData.copy();
    @Nullable Tensor outputTensor = outputTensorList.get(0);
    @Nonnull DeltaSet<UUID> deltaSet = new DeltaSet<>();
    try {
      synchronized (this) {
        if (null != output) {
          output.freeRef();
          output = null;
        }
      }
      output = outputTensor.copy();
      if (isCalcDerivative()) {
        eval.accumulate(deltaSet, getFeedback(outputTensorList));
      }
      return this;
    } finally {
      outputTensor.freeRef();
      evalData.freeRef();
      outputTensorList.freeRef();
      eval.freeRef();
      deltaSet.freeRef();
    }
  }

  @Nullable
  public Tensor[] getDerivative() {
    return derivative;
  }

  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    return TensorArray.wrap(data.stream().map(t -> {
      @Nullable Tensor map = t.map(v -> 1.0);
      t.freeRef();
      return map;
    }).toArray(i -> new Tensor[i]));
  }

  @Nullable
  public Tensor getOutput() {
    return output;
  }

  @Nullable
  public Tensor getOutputAndFree() {
    Tensor output = this.output;
    output.addRef();
    freeRef();
    return output;
  }

  public boolean isCalcDerivative() {
    return calcDerivative;
  }

  public SimpleEval setValidateDerivative(boolean calcDerivative) {
    this.calcDerivative = calcDerivative;
    return this;
  }
}
