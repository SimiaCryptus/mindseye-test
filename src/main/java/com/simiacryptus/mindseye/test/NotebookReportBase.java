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

import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.simiacryptus.aws.S3Util;
import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.extension.AfterTestExecutionCallback;
import org.junit.jupiter.api.extension.BeforeTestExecutionCallback;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.lang.management.ManagementFactory;
import java.net.URI;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Optional;

@ExtendWith(NotebookReportBase.ReportingTestExtension.class)
public abstract class NotebookReportBase {

  protected static final Logger logger = LoggerFactory.getLogger(NotebookReportBase.class);

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  private MarkdownNotebookOutput log;

  protected MarkdownNotebookOutput getLog() {
    return log;
  }

  public Class<? extends NotebookReportBase> getReportClass() {
    return getClass();
  }

  @Nonnull
  public abstract ReportType getReportType();

  protected abstract Class<?> getTargetClass();

  @Nullable
  public static CharSequence setReportType(@Nonnull NotebookOutput log, @Nullable Class<?> networkClass,
                                           final CharSequence prefix) {
    if (null == networkClass)
      return null;
    @Nullable
    String javadoc = CodeUtil.getJavadoc(networkClass);
    log.setMetadata(prefix + "_class_short", networkClass.getSimpleName());
    log.setMetadata(prefix + "_class_full", networkClass.getCanonicalName());
    assert javadoc != null;
    log.setMetadata(prefix + "_class_doc", javadoc.replaceAll("\n", ""));
    return javadoc;
  }

  @NotNull
  public static String toPathString(@Nonnull Class<?> sourceClass) {
    return toPathString(sourceClass, File.separatorChar);
  }

  @NotNull
  public static String toPathString(@Nonnull Class<?> sourceClass, char separatorChar) {
    return sourceClass.getCanonicalName()
        .replace('.', separatorChar)
        .replace('$', separatorChar);
  }

  public void printHeader(@Nonnull NotebookOutput log) {
    log.setMetadata("created_on", new Date().toString());
    log.setMetadata("report_type", getReportType().name());
    CharSequence targetDescription = setReportType(log, getTargetClass(), "network");
    if (null != targetDescription && targetDescription.length() > 0) {
      log.p("__Target Description:__ " + targetDescription);
    }
    CharSequence reportDescription = setReportType(log, getReportClass(), "report");
    if (null != reportDescription && reportDescription.length() > 0) {
      log.p("__Report Description:__ " + reportDescription);
    }
  }

  @AfterEach
  void closeLog() {
    if (null != log) {
      log.close();
      this.log = null;
    }
  }

  @BeforeEach
  void initializeLog(TestInfo testInfo) {
    Class<?> targetClass = getTargetClass();
    @Nonnull
    File reportRoot = new File(Util.mkString(File.separator,
        TestSettings.INSTANCE.testRepo,
        toPathString(targetClass),
        testInfo.getTestClass().map(c1 -> c1.getSimpleName()).orElse(""),
        testInfo.getTestMethod().get().getName(),
        new SimpleDateFormat("yyyyMMddmmss").format(new Date())
    ));
    reportRoot.mkdirs();
    logger.info(RefString.format("Output Location: %s", reportRoot.getAbsoluteFile()));
    if (null != log) throw new IllegalStateException();
    log = new MarkdownNotebookOutput(
        reportRoot, true, testInfo.getTestMethod().get().getName()
    );
    log.setEnableZip(false);
    URI testArchive = TestSettings.INSTANCE.testArchive;
    if (null != testArchive) log.setArchiveHome(testArchive.resolve(
        Util.mkString("/",
            toPathString(targetClass, '/'),
            testInfo.getTestClass().map(c -> c.getSimpleName()).orElse(""),
            testInfo.getTestMethod().get().getName(),
            new SimpleDateFormat("yyyyMMddmmss").format(new Date())
        )
    ));
    S3Util.uploadOnComplete(log, AmazonS3ClientBuilder.standard().build());
    File metadataLocation = new File(TestSettings.INSTANCE.testRepo, "registry");
    metadataLocation.mkdirs();
    log.setMetadataLocation(metadataLocation);
    printHeader(this.log);
  }

  public enum ReportType {
    Applications, Components, Models, Data, Optimizers, Experiments
  }

  static class ReportingTestExtension implements BeforeTestExecutionCallback, AfterTestExecutionCallback {
    private static final String START_TIME = "start time";
    private static final String START_GC_TIME = "start gc time";
    private static final String REFLEAK_MONITOR = "refleak monitor";

    public static ExtensionContext.Store getStore(ExtensionContext context) {
      return context.getStore(ExtensionContext.Namespace.create(ReportingTestExtension.class, context.getRequiredTestMethod()));
    }

    @Override
    public void beforeTestExecution(ExtensionContext context) {
      ExtensionContext.Store store = getStore(context);
      store.put(START_TIME, System.currentTimeMillis());
      store.put(START_GC_TIME, TimedResult.gcTime());
      NotebookReportBase reportingTest = (NotebookReportBase) context.getTestInstance().get();
      store.put(REFLEAK_MONITOR, CodeUtil.refLeakMonitor(reportingTest.getLog()));
    }

    @Override
    public void afterTestExecution(ExtensionContext context) throws Exception {
      System.gc();
      logger.info("Total memory after GC: " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed());
      ExtensionContext.Store store = getStore(context);
      long duration = System.currentTimeMillis() - store.remove(START_TIME, long.class);
      long gcTime = TimedResult.gcTime() - store.remove(START_GC_TIME, long.class);
      NotebookReportBase reportingTest = (NotebookReportBase) context.getTestInstance().get();
      MarkdownNotebookOutput log = reportingTest.getLog();
      log.setMetadata("execution_time", RefString.format("%.3f", duration / 1e3));
      log.setMetadata("gc_time", RefString.format("%.3f", gcTime / 1e3));
      Optional<Throwable> executionException = context.getExecutionException();
      if (executionException.isPresent()) {
        String string = MarkdownNotebookOutput.getExceptionString(executionException.get()).toString();
        string = MarkdownNotebookOutput.replaceAll(string, "\n", "<br/>").trim();
        log.setMetadata("result", string);
      } else {
        log.setMetadata("result", "OK");
      }
      store.remove(REFLEAK_MONITOR, AutoCloseable.class).close();
    }

  }

}
