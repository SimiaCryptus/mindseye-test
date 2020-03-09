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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.data.DoubleStatistics;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class ReferenceIO extends ComponentTestBase<ToleranceStatistics> {
  @Nullable
  final RefHashMap<Tensor[], Tensor> referenceIO;

  public ReferenceIO(@Nullable final RefHashMap<Tensor[], Tensor> referenceIO) {
    this.referenceIO = referenceIO;
  }


  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                  @Nonnull final Tensor... inputPrototype) {
    assert referenceIO != null;
    if (!referenceIO.isEmpty()) {
      log.h1("Reference Input/Output Pairs");
      log.p("Display pre-setBytes input/output example pairs:");
      referenceIO.forEach(RefUtil.wrapInterface((input, output) -> {
        @Nonnull final SimpleEval evalObj = log.eval(() -> {
          @Nonnull final SimpleEval eval = SimpleEval.run(layer.addRef(), RefUtil.addRef(input));
          System.out.println(toString(RefUtil.addRef(input), RefUtil.addRef(output), eval.addRef()));
          return eval;
        });
        verifyNonZero(evalObj);
        RefUtil.freeRef(input);
        RefUtil.freeRef(output);
      }));
    } else {
      log.h1("Example Input/Output Pair");
      log.p("Display input/output pairs from random executions:");
      @Nonnull final SimpleEval evalObj = log.eval(() -> {
        @Nonnull final SimpleEval eval = SimpleEval.run(layer.addRef(), RefUtil.addRef(inputPrototype));
        System.out.println(toString(RefUtil.addRef(inputPrototype), eval.addRef()));
        return eval;
      });
      verifyNonZero(evalObj);
    }
    layer.freeRef();
    RefUtil.freeRef(inputPrototype);
    return null;
  }

  @Nonnull
  @Override
  public String toString() {
    return "ReferenceIO{" + "referenceIO=" + referenceIO + '}';
  }

  public void _free() {
    if (null != referenceIO)
      referenceIO.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ReferenceIO addRef() {
    return (ReferenceIO) super.addRef();
  }

  private void verifyNonZero(SimpleEval eval) {
    Tensor output = eval.getOutput();
    double rms = output.rms();
    output.freeRef();
    eval.freeRef();
    if (rms == 0) {
      throw new AssertionError();
    }
  }

  @NotNull
  private String toString(@Nonnull Tensor[] input, SimpleEval eval) {
    Tensor evalOutput = eval.getOutput();
    try {
      return RefString.format(
          "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\n--------------------\nDerivative: \n%s",
          RefArrays.stream(input).map(t -> {
            String temp_05_0006 = t.prettyPrint();
            t.freeRef();
            return temp_05_0006;
          }).reduce((a, b) -> a + ",\n" + b).orElse(""), RefArrays.toString(evalOutput.getDimensions()),
          evalOutput.prettyPrint(), RefArrays.stream(eval.getDerivative()).map(t -> {
            String temp_05_0007 = t.prettyPrint();
            t.freeRef();
            return temp_05_0007;
          }).reduce((a, b) -> a + ",\n" + b).orElse(""));
    } finally {
      if (null != evalOutput) evalOutput.freeRef();
      eval.freeRef();
    }
  }

  @NotNull
  private String toString(Tensor[] input, Tensor output, SimpleEval eval) {
    Tensor evalOutput = eval.getOutput();
    Tensor difference = evalOutput == null ? null : Tensor.add(output.scale(-1), evalOutput.addRef());
    output.freeRef();
    @Nonnull final DoubleStatistics error = difference.getDoubleStatistics();
    difference.freeRef();
    try {
      return RefString.format(
          "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\nError: %s\n--------------------\nDerivative: \n%s",
          RefUtil.get(RefArrays.stream(input).map(t -> {
            String s = RefArrays.toString(t.getDimensions()) + "\n" + t.prettyPrint();
            t.freeRef();
            return s;
          }).reduce((a, b) -> a + ",\n" + b)), RefArrays.toString(evalOutput.getDimensions()),
          evalOutput.prettyPrint(), error, RefUtil.get(RefArrays.stream(eval.getDerivative()).map(t -> {
            String s = t.prettyPrint();
            t.freeRef();
            return s;
          }).reduce((a, b) -> a + ",\n" + b)));
    } finally {
      RefUtil.freeRef(eval);
      evalOutput.freeRef();
    }
  }
}
