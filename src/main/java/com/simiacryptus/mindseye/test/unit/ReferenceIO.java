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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.util.data.DoubleStatistics;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public @RefAware
class ReferenceIO extends ComponentTestBase<ToleranceStatistics> {
  final RefHashMap<Tensor[], Tensor> referenceIO;

  public ReferenceIO(final RefHashMap<Tensor[], Tensor> referenceIO) {
    this.referenceIO = referenceIO;
  }

  public static @SuppressWarnings("unused")
  ReferenceIO[] addRefs(ReferenceIO[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReferenceIO::addRef)
        .toArray((x) -> new ReferenceIO[x]);
  }

  public static @SuppressWarnings("unused")
  ReferenceIO[][] addRefs(ReferenceIO[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReferenceIO::addRefs)
        .toArray((x) -> new ReferenceIO[x][]);
  }

  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                  @Nonnull final Tensor... inputPrototype) {
    if (!referenceIO.isEmpty()) {
      log.h1("Reference Input/Output Pairs");
      log.p("Display pre-setBytes input/output example pairs:");
      referenceIO.forEach((input, output) -> {
        log.eval(() -> {
          @Nonnull final SimpleEval eval = SimpleEval.run(layer, input);
          Tensor evalOutput = eval.getOutput();
          Tensor difference = output.scale(-1).addAndFree(evalOutput);
          @Nonnull final DoubleStatistics error = new DoubleStatistics().accept(difference.getData());
          return String.format(
              "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\nError: %s\n--------------------\nDerivative: \n%s",
              RefArrays.stream(input)
                  .map(
                      t -> RefArrays.toString(t.getDimensions()) + "\n" + t.prettyPrint())
                  .reduce((a, b) -> a + ",\n" + b).get(),
              RefArrays.toString(evalOutput.getDimensions()), evalOutput.prettyPrint(),
              error, RefArrays.stream(eval.getDerivative()).map(t -> t.prettyPrint())
                  .reduce((a, b) -> a + ",\n" + b).get());
        });
      });
    } else {
      log.h1("Example Input/Output Pair");
      log.p("Display input/output pairs from random executions:");
      log.eval(() -> {
        @Nonnull final SimpleEval eval = SimpleEval.run(layer, inputPrototype);
        Tensor evalOutput = eval.getOutput();
        return String.format(
            "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\n--------------------\nDerivative: \n%s",
            RefArrays.stream(inputPrototype).map(t -> t.prettyPrint())
                .reduce((a, b) -> a + ",\n" + b).orElse(""),
            RefArrays.toString(evalOutput.getDimensions()), evalOutput.prettyPrint(),
            RefArrays.stream(eval.getDerivative()).map(t -> t.prettyPrint())
                .reduce((a, b) -> a + ",\n" + b).orElse(""));
      });
    }
    return null;
  }

  @Nonnull
  @Override
  public String toString() {
    return "ReferenceIO{" + "referenceIO=" + referenceIO + '}';
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  ReferenceIO addRef() {
    return (ReferenceIO) super.addRef();
  }
}
