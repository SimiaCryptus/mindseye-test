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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.function.IntFunction;

/**
 * The type Simple list eval.
 */
public class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
  @Nonnull
  private final TensorList[] input;
  @Nonnull
  private final Layer layer;
  private boolean calcDerivatives = true;
  @Nullable
  private TensorList[] inputDerivative;
  @Nullable
  private TensorList output;
  @Nullable
  private DeltaSet<UUID> layerDerivative;

  /**
   * Instantiates a new Simple list eval.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleListEval(@Nonnull final Layer layer, @Nonnull final TensorList... input) {
    Layer temp_09_0001 = layer.addRef();
    this.layer = temp_09_0001 == null ? null : temp_09_0001.addRef();
    if (null != temp_09_0001)
      temp_09_0001.freeRef();
    layer.freeRef();
    TensorList[] temp_09_0002 = RefUtil.addRef(input);
    this.input = RefUtil.addRef(temp_09_0002);
    RefUtil.freeRef(temp_09_0002);
    RefUtil.freeRef(input);
    DeltaSet<UUID> temp_09_0003 = new DeltaSet<UUID>();
    layerDerivative = temp_09_0003.addRef();
    temp_09_0003.freeRef();
  }

  @Nullable
  @Override
  public TensorList[] getInputDerivative() {
    assertAlive();
    return RefUtil.addRef(inputDerivative);
  }

  @Nullable
  public DeltaSet<UUID> getLayerDerivative() {
    return layerDerivative == null ? null : layerDerivative.addRef();
  }

  /**
   * Sets layer derivative.
   *
   * @param layerDerivative the layer derivative
   */
  public void setLayerDerivative(@Nullable DeltaSet<UUID> layerDerivative) {
    DeltaSet<UUID> temp_09_0004 = layerDerivative == null ? null : layerDerivative.addRef();
    if (null != this.layerDerivative)
      this.layerDerivative.freeRef();
    this.layerDerivative = temp_09_0004 == null ? null : temp_09_0004.addRef();
    if (null != temp_09_0004)
      temp_09_0004.freeRef();
    if (null != layerDerivative)
      layerDerivative.freeRef();
  }

  @Nullable
  @Override
  public TensorList getOutput() {
    return output == null ? null : output.addRef();
  }

  /**
   * Is calc derivatives boolean.
   *
   * @return the boolean
   */
  public boolean isCalcDerivatives() {
    return calcDerivatives;
  }

  /**
   * Sets calc derivatives.
   *
   * @param calcDerivatives the calc derivatives
   */
  public void setCalcDerivatives(boolean calcDerivatives) {
    this.calcDerivatives = calcDerivatives;
  }

  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public static void accumulate(@Nonnull final TensorList buffer, @Nonnull final TensorList data) {
    RefIntStream.range(0, data.length()).forEach(RefUtil.wrapInterface(b -> {
      @Nullable
      Tensor r = data.get(b);
      @Nullable
      Tensor l = buffer.get(b);
      l.addInPlace(r.addRef());
      l.freeRef();
      r.freeRef();
    }, data, buffer));
  }

  /**
   * Run simple result.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple result
   */
  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, @Nullable final TensorList... tensor) {
    return run(layer, true, tensor);
  }

  /**
   * Run simple result.
   *
   * @param layer           the layer
   * @param calcDerivatives the calc derivatives
   * @param tensor          the tensor
   * @return the simple result
   */
  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, boolean calcDerivatives, @Nullable final TensorList... tensor) {
    SimpleListEval eval = new SimpleListEval(layer, tensor);
    eval.setCalcDerivatives(calcDerivatives);
    SimpleResult temp_09_0013 = eval.call();
    eval.freeRef();
    return temp_09_0013;
  }

  @Nonnull
  @Override
  public SimpleResult call() {
    TensorList[] inputCopy = RefArrays.stream(RefUtil.addRef(input)).map(tensorList -> {
      TensorList copy = tensorList.copy();
      tensorList.freeRef();
      return copy;
    }).toArray(TensorList[]::new);
    if (null != inputDerivative)
      RefUtil.freeRef(inputDerivative);
    inputDerivative = RefArrays.stream(RefUtil.addRef(inputCopy)).map(tensorList -> {
      TensorArray tensorArray = new TensorArray(tensorList.stream().map(tensor -> {
        int[] dimensions = tensor.getDimensions();
        tensor.freeRef();
        return new Tensor(dimensions);
      }).toArray(Tensor[]::new));
      tensorList.freeRef();
      return tensorArray;
    }).toArray(TensorList[]::new);
    Result[] inputs = RefIntStream.range(0, inputCopy.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Result>) i -> {
          Result.Accumulator accumulator = new Result.Accumulator() {
            {
              RefUtil.addRef(inputDerivative);
            }

            @Override
            public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
              buffer.freeRef();
              SimpleListEval.accumulate(inputDerivative[i].addRef(), data);
            }

            @Override
            public void _free() {
              RefUtil.freeRef(inputDerivative);
              super._free();
            }
          };
          return new Result(inputCopy[i].addRef(), accumulator, true);
        }, inputCopy)).toArray(Result[]::new);
    @Nullable final Result eval = layer.eval(inputs);
    TensorList data = eval.getData();
    TensorList outputData = data.copy();
    data.freeRef();
    if (null != this.layerDerivative)
      this.layerDerivative.freeRef();
    this.layerDerivative = new DeltaSet<>();
    if (isCalcDerivatives())
      eval.accumulate(layerDerivative == null ? null : layerDerivative.addRef(),
          getFeedback(outputData == null ? null : outputData.addRef()));
    eval.freeRef();
    if (null != output)
      output.freeRef();
    output = outputData;
    return this.addRef();
  }

  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    TensorArray temp_09_0014 = new TensorArray(data.stream().map(t -> {
      Tensor temp_09_0011 = t.map(v -> 1.0);
      t.freeRef();
      return temp_09_0011;
    }).toArray(Tensor[]::new));
    data.freeRef();
    return temp_09_0014;
  }

  public void _free() {
    super._free();
    if (null != layerDerivative)
      layerDerivative.freeRef();
    layerDerivative = null;
    if (null != output)
      output.freeRef();
    output = null;
    if (null != inputDerivative)
      RefUtil.freeRef(inputDerivative);
    inputDerivative = null;
    layer.freeRef();
    RefUtil.freeRef(input);
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SimpleListEval addRef() {
    return (SimpleListEval) super.addRef();
  }
}
