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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.concurrent.Callable;

/**
 * The type Simple eval.
 */
public class SimpleEval extends ReferenceCountingBase implements Callable<SimpleEval> {
  @Nonnull
  private final Tensor[] input;
  @Nonnull
  private final Layer layer;
  @Nullable
  private final Tensor[] derivative;
  private boolean calcDerivative = false;
  @Nullable
  private Tensor output;

  /**
   * Instantiates a new Simple eval.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleEval(@Nonnull final Layer layer, @Nonnull final Tensor... input) {
    this.layer = layer;
    this.input = input;
    this.output = null;
    this.derivative = RefArrays.stream(RefUtil.addRef(this.input)).map(tensor -> {
      try {
        return tensor.getDimensions();
      } finally {
        tensor.freeRef();
      }
    }).map(dims -> new Tensor(dims)).toArray(value -> new Tensor[value]);
  }

  /**
   * Get derivative tensor [ ].
   *
   * @return the tensor [ ]
   */
  @Nullable
  @JsonIgnore
  public Tensor[] getDerivative() {
    return RefUtil.addRef(derivative);
  }

  /**
   * Gets output.
   *
   * @return the output
   */
  @Nullable
  public Tensor getOutput() {
    return output == null ? null : output.addRef();
  }

  /**
   * Is calc derivative boolean.
   *
   * @return the boolean
   */
  public boolean isCalcDerivative() {
    return calcDerivative;
  }

  /**
   * Sets result.
   *
   * @param eval the eval
   */
  public void setResult(Result eval) {
    assert eval != null;
    try {
      TensorList evalData = eval.getData();
      Tensor outputTensor = evalData.get(0);
      Tensor copy = outputTensor.copy();
      outputTensor.freeRef();
      synchronized (this) {
        if (null != output) {
          output.freeRef();
        }
        output = copy;
      }
      if (isCalcDerivative()) {
        checkedFeedback(eval, evalData);
      } else {
        evalData.freeRef();
      }
    } finally {
      eval.freeRef();
    }
  }

  /**
   * Sets validate derivative.
   *
   * @param calcDerivative the calc derivative
   */
  public void setValidateDerivative(boolean calcDerivative) {
    this.calcDerivative = calcDerivative;
  }

  /**
   * Run simple eval.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple eval
   */
  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, @Nullable final Tensor... tensor) {
    return run(layer, true, tensor);
  }

  /**
   * Run simple eval.
   *
   * @param layer              the layer
   * @param validateDerivative the validate derivative
   * @param tensor             the tensor
   * @return the simple eval
   */
  @Nonnull
  public static SimpleEval run(@Nonnull final Layer layer, boolean validateDerivative, @Nullable final Tensor... tensor) {
    SimpleEval simpleEval = new SimpleEval(layer, tensor);
    simpleEval.setValidateDerivative(validateDerivative);
    simpleEval.eval();
    return simpleEval;
  }

  @Nonnull
  @Override
  public SimpleEval call() {
    eval();
    return this.addRef();
  }

  /**
   * Eval.
   */
  public void eval() {
    setResult(layer.eval(input()));
  }

  /**
   * Input result [ ].
   *
   * @return the result [ ]
   */
  @NotNull
  public Result[] input() {
    return RefIntStream.range(0, input.length).mapToObj(i -> {
      Result.Accumulator accumulator = new Accumulator(derivative[i].addRef());
      TensorArray data = new TensorArray(input[i].copy());
      return new Result(data, accumulator, true);
    }).toArray(Result[]::new);
  }

  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
    try {
      return new TensorArray(data.stream().map(t -> {
        try {
          return t.map(v -> 1.0);
        } finally {
          t.freeRef();
        }
      }).toArray(Tensor[]::new));
    } finally {
      data.freeRef();
    }
  }

  public void _free() {
    super._free();
    RefUtil.freeRef(derivative);
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

  @RefIgnore
  private void checkedFeedback(@RefIgnore Result eval, TensorList evalData) {
    TensorList feedback = getFeedback(evalData);
    eval.accumulate(new DeltaSet<>(), feedback);
    if (!feedback.isFreed()) {
      throw new IllegalStateException("Feedback TensorList not freed after accumulate");
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private Tensor tensor;

    /**
     * Instantiates a new Accumulator.
     *
     * @param tensor the tensor
     */
    public Accumulator(Tensor tensor) {
      this.tensor = tensor;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      try {
        data.stream().forEach(t -> {
          tensor.addInPlace(t);
        });
      } finally {
        buffer.freeRef();
        data.freeRef();
      }
    }

    @Override
    public void _free() {
      RefUtil.freeRef(tensor);
      super._free();
    }
  }
}
