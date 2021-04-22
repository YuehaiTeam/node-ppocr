{
    "targets": [
        {
            "target_name": "ppocr",
            "cflags!": ["-fno-exceptions"],
            "cflags_cc!": ["-fno-exceptions"],
            "sources": [
                "ppocr.cc",
                "includes/paddle/clipper.cpp",
                "includes/paddle/ocr_det.cpp",
                "includes/paddle/ocr_rec.cpp",
                "includes/paddle/postprocess_op.cpp",
                "includes/paddle/preprocess_op.cpp",
                "includes/paddle/utility.cpp",
            ],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "includes"
            ],
            "libraries": [
                "<(module_root_dir)/lib/paddle_inference.lib",
                "<(module_root_dir)/lib/opencv_core452.lib",
                "<(module_root_dir)/lib/opencv_imgproc452.lib"
            ],
            'defines': ['NAPI_DISABLE_CPP_EXCEPTIONS']
        }
    ]
}
