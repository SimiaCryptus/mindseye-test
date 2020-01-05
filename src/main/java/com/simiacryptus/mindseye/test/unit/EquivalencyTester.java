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

import com.google.gson.GsonBuilder;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public @RefAware
class EquivalencyTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);

  private final Layer reference;
  private final double tolerance;

  public EquivalencyTester(final double tolerance, final Layer referenceLayer) {
    this.tolerance = tolerance;
    this.reference = referenceLayer;
  }

  public static @SuppressWarnings("unused")
  EquivalencyTester[] addRefs(EquivalencyTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EquivalencyTester::addRef)
        .toArray((x) -> new EquivalencyTester[x]);
  }

  public static @SuppressWarnings("unused")
  EquivalencyTester[][] addRefs(EquivalencyTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EquivalencyTester::addRefs)
        .toArray((x) -> new EquivalencyTester[x][]);
  }

  public ToleranceStatistics test(@Nullable final Layer subject, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference || null == subject)
      return new ToleranceStatistics();
    reference.assertAlive();
    final Tensor subjectOutput = SimpleEval.run(subject, inputPrototype).getOutput();
    final Tensor referenceOutput = SimpleEval.run(reference, false, inputPrototype).getOutput();
    @Nonnull
    Tensor error = null;
    {
      log.info(String.format("Inputs: %s", RefArrays.stream(inputPrototype)
          .map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
      log.info(String.format("Subject Output: %s", subjectOutput.prettyPrint()));
      log.info(String.format("Reference Output: %s", referenceOutput.prettyPrint()));
      error = subjectOutput.minus(referenceOutput);
      log.info(String.format("Error: %s", error.prettyPrint()));
      @Nonnull final ToleranceStatistics result = RefIntStream.range(0, subjectOutput.length())
          .mapToObj(i1 -> {
            return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
          }).reduce((a, b) -> a.combine(b)).get();
      log.info(String.format("Accuracy:"));
      log.info(String.format("absoluteTol: %s", result.absoluteTol.toString()));
      log.info(String.format("relativeTol: %s", result.relativeTol.toString()));
      if (!(result.absoluteTol.getMax() < tolerance))
        throw new AssertionError(result.toString());
      return result;
    }
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, final Layer subject,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Reference Implementation");
    output.p("This key is an alternate implementation which is expected to behave the same as the following key:");
    output.run(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(reference.getJson()));
    });
    output.run(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(subject.getJson()));
    });
    output.p("We measureStyle the agreement between the two layers in a random execution:");
    return output.eval(() -> {
      return test(subject, inputPrototype);
    });
  }

  @Nonnull
  @Override
  public String toString() {
    return "EquivalencyTester{" + "reference=" + reference + ", tolerance=" + tolerance + '}';
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  EquivalencyTester addRef() {
    return (EquivalencyTester) super.addRef();
  }
}
