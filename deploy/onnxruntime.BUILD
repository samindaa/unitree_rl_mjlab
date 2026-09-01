load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_import(
    name = "onnxruntime_import",
    shared_library = "lib/libonnxruntime.so.1",
)

cc_library(
    name = "onnxruntime",
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":onnxruntime_import"],
)
